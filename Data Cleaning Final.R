library("tidyverse")
library("lubridate")
library("dplyr")
library("ggplot2")
library("tseries")
library("urca")

setwd("path-to-data-folder")

# ------------------------- Assets Data <- ----------------------------------
raw_assets <- "BHCK Series 2 Q12022 to Q42023.csv"
link_series <- "2022Q3 BankID-Permco Link.csv"
df <- read.csv(raw_assets)
link_permco <- read.csv(link_series)

# rssd9001 RSSD ID
# rssd9017 Legal name
# rssd9999 Date
# bhck2948 Total liabilities and minority interest
# bhck2170 Total assets 

df_final <- df %>% filter(rssd9999=="30/09/2022") %>% arrange(desc(bhck2170)) %>% 
  select(rssd9001, rssd9017, rssd9999, bhck2948, bhck2170) %>%
  distinct(rssd9001, .keep_all = TRUE) %>%
  drop_na(bhck2170)


linked_df <- df_final %>%
  inner_join(link_permco, by = c("rssd9001" = "entity"))

write_csv(linked_df, "largest_banks_q12023.csv")

linked_df_small <- linked_df[1:102,]


# ---------------------------- Import Linking Sheet for 100 bank subset -------

link = read_csv("permco_cusip link.csv")
# Pre sorted by total assets TOTAST
link <- link[1:102,]
link <- link %>% rename(PERMCO = permco, TOTAST = totast)

# ------------------------- START: Stock Data ----------------------------------
start_date <- as.Date("2022-08-15") # Start of window
end_date <- as.Date("2022-11-14") # End of window 

# Variables 
# comnam Company Name
# ticker Ticker
# permco Permco number
# prc Bid/Ask average
# vol Volume ()
# shrout Number of Shares outstanding ()


stock_series <- "crsp daily stocks by permco jan22-dec23.csv"
df_stock_raw <- read_csv(stock_series)
df_stock_raw$date <- as.Date(df_stock_raw$date, format = "%d/%m/%Y")
df_stock_raw$SHROUT <- df_stock_raw$SHROUT*1000 # Convert from 1000s to units

# Convert prices to equity value of firm
df_stock_raw$VAL <- df_stock_raw$PRC * df_stock_raw$SHROUT 


df_stock <- df_stock_raw %>% filter(date >= start_date & date <= end_date)

df_stock <- df_stock %>% select("PERMCO", "date", "CUSIP", "COMNAM", "TICKER", "PRC", "SHROUT", "VAL")

# print(count(df_stock, TICKER), n=102) # Check there are the same number of trading days in each stock

# XLookup and join the linker TOTASTs by permco, remove NAs (banks ranked below 100ish)
# df_stock <- left_join(df_stock, link %>% select(PERMCO, TOTAST), by="PERMCO") %>% 
  # drop_na()


# Get value cols only
df_stock_w <- df_stock %>% arrange(desc(VAL)) %>% select(CUSIP, VAL, date)
# Pivot 
df_stock_w <- df_stock_w %>%
  group_by(date, CUSIP) %>% 
  summarise(VAL = mean(VAL, na.rm=TRUE)) %>% 
  ungroup %>% 
  pivot_wider(names_from = CUSIP, values_from = VAL) %>% 
  select(-date)
df_stock_w <- df_stock_w[, c("46625H10",	"17296742",	"94974610",	"38141G10",	"61744644",	"90297330",	"80851310",	"69347510",	"89832Q10",	"14040H10")]

unit_means <- colMeans(df_stock_w, na.rm = TRUE)  # Mean across time for each unit
df_stock_mean <- data.frame(
  cusip = names(unit_means),
  mean = as.numeric(unit_means)
)

df_stock_w <- log(df_stock_w)
# Compute logged covariance matrix
vcov_matrix <- cov(df_stock_w)
vcov_df <- as.data.frame(as.table(vcov_matrix)) # Proper dataframe where (i.j) is cov(i,j)
isSymmetric(vcov_matrix) # Check symmetric

vcov_df <- vcov_df %>% rename(cusip1 = Var1, cusip2 = Var2, covariance = Freq)

# -------------------------- Options Data -------------------------------

# CUSIP to link

options_small = "OptionsMetrics Call Only Jan22-Aug23 SMALL.csv"

df_os = read_csv(options_small)

# Options data is way better
# What is the units of strike price?
# Need to get annual risk free rate


# Filter for all options that have expiration date between Mon 12th Sept 2022 and Fri 14th Oct 2022 -- roughly around Q3.

df_ost <- df_os
df_ost$exdate <- as.Date(df_ost$exdate, format = "%d/%m/%Y")
df_ost$date <- as.Date(df_ost$date, format = "%d/%m/%Y")

df_os_new <- df_ost %>% filter(exdate >= start_date & exdate <= end_date)

df_os_new$extime <- as.numeric(df_os_new$exdate - df_os_new$date)/365 # Fraction of year left until expiration
df_os_new$strike_price <- df_os_new$strike_price/1000 # Data is in 10ths of cents, convert back to dollar units

df_os_new <- df_os_new %>% select(ticker, cusip, date, exdate, extime, strike_price, impl_volatility)

# check = df_os_new %>% drop_na() %>% distinct(cusip, .keep_all = TRUE) # Check how many banks have impl volatilities calculated

# Merge underlying stock prices at date (t=0) 

df_os_merged <- left_join(df_os_new, df_stock_raw %>% select(CUSIP, date, PRC, SHROUT) %>% rename(cusip=CUSIP), by=c("cusip", "date"))

df_os_merged <- df_os_merged %>% drop_na()

# It's better to use forward prices since these stocks pay dividends:

df_forward <- read_csv("optionsmetrics forward_prices jan22-aug23.csv")
df_forward$date <- as.Date(df_forward$date, format = "%d/%m/%Y")
df_forward$expiration <- as.Date(df_forward$expiration, format = "%d/%m/%Y")

df_os_merged <- left_join(df_os_merged, df_forward %>% select(cusip, date, expiration, ForwardPrice) %>% rename(exdate=expiration), by=c("cusip", "date", "exdate"))

risk_neutral_prob_ITM <- function(forward, K, time, sigma) {
  d2 <- (log(forward / K) + (0.5 * sigma^2) * T) / (sigma * sqrt(time))
  p_ITM <- pnorm(d2)  # CDF of standard normal
  return(p_ITM)
  
}

df_os_merged$prob_ITM = risk_neutral_prob_ITM(df_os_merged$ForwardPrice,
                                              df_os_merged$strike_price,
                                              df_os_merged$extime,
                                              df_os_merged$impl_volatility)

# Now we have many trading days of ITM probabilities
# Convert strike price to strike value, ensure shares outstanding are same for all dates by just taking average
# Accounts for share issuances 

df_os_merged_avg_SHROUT <- df_os_merged %>% 
  group_by(cusip) %>%
  mutate(SHROUT=mean(SHROUT, na.rm=TRUE)) %>% 
  ungroup()

df_os_merged_avg_SHROUT$strike_value = df_os_merged_avg_SHROUT$strike_price * df_os_merged_avg_SHROUT$SHROUT

average_probabilities <- df_os_merged_avg_SHROUT %>%
  select(ticker, cusip, strike_value, prob_ITM) %>% 
  group_by(ticker, cusip, strike_value) %>%
  summarise(mean_prob = mean(prob_ITM, na.rm = TRUE), .groups = 'drop')

# ------------------------ CDS Data -------------------------------
# NO PERMCO -- Need to link by ticker

linker = "permco_cusip_redcode_link.csv"
cds = "markit_cds.csv"
df_link = read_csv(linker) %>% drop_na(REDCODE)
df_cds = read_csv(cds)

df_cds_test <- df_cds 
df_cds_test <- df_cds_test %>% filter(tenor%in% c("3M", "6M", "1Y"), tier=="SNRFOR", currency=="USD")
df_cds_test$date <- as.Date(df_cds_test$date, format="%d/%m/%Y")
df_cds_test$enddate <- ifelse(df_cds_test$tenor == "3M",
                              df_cds_test$date %m+% months(3),
                              ifelse(df_cds_test$tenor == "6M",
                                     df_cds_test$date %m+% months(6),
                                     ifelse(df_cds_test$tenor == "1Y",
                                            df_cds_test$date %m+% years(1),
                                            NA)))
df_cds_test$maturity <- ifelse(df_cds_test$tenor == "3M",
                               0.25,
                               ifelse(df_cds_test$tenor == "6M",
                                      0.5,
                                      ifelse(df_cds_test$tenor == "1Y",
                                             1,
                                             NA)))

df_cds_test$enddate <- as.Date(df_cds_test$enddate)
df_cds_test <- df_cds_test %>% filter(enddate >= start_date & enddate <=end_date)
tickers2 = df_cds_test %>% distinct(ticker, .keep_all = TRUE)


calc_pd <- function(spread, recovery, maturity_years) {
  # Calculate cumulative probability of default
  pd <- 1 - exp(-(spread / (1 - recovery)) * maturity_years)
  
  return(pd)
}

df_cds_test$prob_def <- calc_pd(df_cds_test$parspread, recovery=0.4, df_cds_test$maturity)
df_cds_test <- left_join(df_cds_test, df_link %>% select(CUSIP, REDCODE) %>% rename(redcode=REDCODE, cusip=CUSIP), by="redcode")
df_cds_test <- df_cds_test %>% select(ticker, shortname,cusip, date, enddate, maturity, prob_def)

avg_pd <-df_cds_test %>%
  group_by(ticker, cusip) %>%
  summarise(mean_prob = mean(prob_def, na.rm = TRUE), .groups = 'drop')


# ----------------------- Make consistent and write ---------------------------

permco_link <- read_csv("permco_cusip link.csv")

common_vals <- df_stock_mean$cusip %>% intersect(avg_pd$cusip)

df_stock_mean_f <- df_stock_mean %>% filter(cusip %in% common_vals)
average_probabilities_f <- average_probabilities %>% filter(cusip %in% common_vals)

vcov_df_f <- vcov_df %>% filter(cusip1 %in% common_vals) %>% filter(cusip2 %in% common_vals)

link_f <-link %>%  select(cusip, TOTAST) %>% filter(cusip %in% common_vals)

avg_pd_f <- avg_pd %>% filter(cusip %in% common_vals)
permco_link <- permco_link %>% filter(cusip %in% common_vals)

write_csv(df_stock_mean_f, "Clean Data/Mean_Prices.csv")
write_csv(vcov_df_f, "Clean Data/Variance_Covariance_Log_Prices.csv")
write_csv(average_probabilities_f, "Clean Data/Call_Strikes_And_ITMProbabilities.csv")
write_csv(link_f, "Clean Data/Total_Assets.csv")
write_csv(avg_pd, "Clean Data/CDS_PD.csv")
write_csv(permco_link, "Clean Data/Permco_Cusip_Link.csv")

# ------------------------------- Stock Stationarity ----------------------------

final_vals = list("46625H10",	"17296742",	"94974610",	"38141G10",	"61744644",	"90297330",	"80851310",	"69347510",	"89832Q10",	"14040H10")
df_stock_w <- exp(df_stock_w)
df_stock_wf <- df_stock_w[, names(df_stock_w) %in% final_vals]
df_stock_wf <- df_stock_wf %>% rename(JPM = `46625H10`,
                                      CITI = `17296742`,
                                      WFC = `94974610`,
                                      GS = `38141G10`,
                                      MS = `61744644`,
                                      USB = `90297330`,
                                      SCH = `80851310`,
                                      PNC = `69347510`,
                                      TFIN = `89832Q10`,
                                      COF = `14040H10`)
df_stock_wf <- df_stock_wf/10^9
df_stock_wf <- df_stock_wf[, c("JPM", "CITI", "WFC", "GS", "MS", "USB", "SCH", "PNC", "TFIN", "COF")]
df_long <- df_stock_wf %>%
  mutate(Time = row_number()) %>%
  pivot_longer(cols = -Time, names_to = "Series", values_to = "Value")

# Plot
ggplot(df_long, aes(x = Time, y = Value, color = Series)) +
  geom_line(size = 1) +
  geom_vline(xintercept = 35, linetype = "dashed", color = "black") +
  annotate("text", x = 35, y = max(df_long$Value, na.rm = TRUE), 
           label = "Q1 2023 (31/03/23)", hjust = -0.1, vjust = 8.5) +
  geom_vline(xintercept = 20, linetype = "dashed", color = "black") +
  annotate("text", x = 20, y = max(df_long$Value, na.rm = TRUE), 
           label = "SVB Failure (10/03/23)", hjust = 1.05, vjust = 6.5) +
  labs(
    title="Evolution of equity value for US bank holding companies around Q1 2023",
    x = "Trading days",
    y = "Equity value (USD bn)",
    color = "Bank HCs"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

test_stationarity <- function(series) {
  adf_result <- adf.test(series, alternative = "stationary")
  kpss_result <- kpss.test(series)
  
  return(list(
    ADF_p_value = adf_result$p.value,
    KPSS_p_value = kpss_result$p.value
  ))
}

# Apply to all variables
results <- lapply(df_stock_wf, test_stationarity)

# Convert to data.frame for readability
stationarity_results <- do.call(rbind, lapply(results, function(x) unlist(x)))
rownames(stationarity_results) <- colnames(df_stock_wf)
stationarity_results
