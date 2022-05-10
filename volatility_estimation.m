% Set the path for the toolbox
addpath("Quantitative Finance\Thesis\Code\mfe-toolbox-main\realized\")
warning("off", "all")

format long g

% User inputs
n_decimals = 7;
start_time = 37800;
end_time   = 55800;
sample_len = 5 * 60;
days       = ["02", "05", "06", "07", "08", "09", "12", "13", "14", "15", "16", ...
    "20", "21", "22", "23", "26", "27", "28", "29", "30"];

% Variables init
n_days = length(days);

real_var       = zeros(1, n_days);
real_ker       = zeros(1, n_days);
real_mult_ker  = zeros(1, n_days);
twoscale       = zeros(1, n_days);
multiscale     = zeros(1, n_days);
real_bipow_var = zeros(1, n_days);
med_real_var   = zeros(1, n_days);
preavg         = zeros(1, n_days);

for i = 1:n_days

    fprintf("2015-01-" + days(i) + "\n")

    % Import order book data
    order_book = importdata("Quantitative Finance\Thesis\Code\data\csv\TSLA_2015-01-" + days(i) + "_34200000_57600000_orderbook_10.csv");
    order_flow = importdata("Quantitative Finance\Thesis\Code\data\csv\TSLA_2015-01-" + days(i) + "_34200000_57600000_message_10.csv");
    
    % Define best bid and best ask prices
    best_bid = order_book(:, 3) / 10000;
    best_ask = order_book(:, 1) / 10000;
    
    % Compute the mid-price
    mid_price = (best_ask + best_bid) / 2;
    mid_price_df = table(order_flow(:, 1), mid_price);
    mid_price_df.Properties.VariableNames = ["time", "price"];
    
    % Compute realized variance
    real_var(i) = realized_variance(mid_price_df.price, mid_price_df.time, ...
        "seconds", "Fixed", start_time:sample_len:end_time);
    real_var(i) = round(real_var(i), n_decimals);
    
    % Compute realized kernel
    options = realized_options('Kernel');
    real_ker(i) = realized_kernel(mid_price_df.price, mid_price_df.time, ...
        "seconds", "Fixed", start_time:sample_len:end_time, options);
    real_ker(i) = round(real_ker(i), n_decimals);
    
    % Compute multivariate realized kernel
    options = realized_options('Multivariate Kernel');
    real_mult_ker(i) = realized_kernel(mid_price_df.price, mid_price_df.time, ...
        "seconds", "Fixed", start_time:sample_len:end_time, options);
    real_mult_ker(i) = round(real_mult_ker(i), n_decimals);
    
    % Compute two-scales realized variance
    twoscale(i) = realized_twoscale_variance(mid_price_df.price);
    twoscale(i) = round(twoscale(i), n_decimals);
    
    % Compute multi-scales estimator
    multiscale(i) = realized_multiscale_variance(mid_price_df.price);
    multiscale(i) = round(multiscale(i), n_decimals);

    % Compute realized bipower variation
    real_bipow_var(i) = realized_bipower_variation(mid_price_df.price, mid_price_df.time, ...
        "seconds", "Fixed", start_time:sample_len:end_time);
    real_bipow_var(i) = round(real_bipow_var(i), n_decimals);
    
    % Compute median realized variance
    med_real_var(i) = realized_min_med_variance(mid_price_df.price, mid_price_df.time, ...
        "seconds", "Fixed", start_time:sample_len:end_time);
    med_real_var(i) = round(med_real_var(i), n_decimals);
    
    % Compute pre-averaging estimator
    preavg(i) = realized_preaveraged_variance(mid_price_df.price, mid_price_df.time, ...
        "seconds", "Fixed", start_time:sample_len:end_time);
    preavg(i) = round(preavg(i), n_decimals);

end

real_var_est = [real_var; real_ker; real_mult_ker; twoscale; multiscale; ...
    real_bipow_var; med_real_var; preavg].';
real_var_est = array2table(real_var_est);
real_var_est.Properties.VariableNames = ["Realized variance", "Realized kernel", ...
    "Multivariate realized kernel", "Two-scales", "Multi-scales", ...
    "Bipower", "Median", "Pre-averaging"];

table2latex(real_var_est)
stackedplot(real_var_est)

