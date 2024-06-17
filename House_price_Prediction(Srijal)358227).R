#Loading necessary libraries
library(dplyr)
library(tidyverse)
library(caret)
library(lattice)
library(corrplot)
library(ggplot2)
library(xgboost)
library(randomForest)
library(tidyr)


#Loading the dataset
df <- read.csv("C:/Users/DELL/Downloads/house_price/houseprice.csv")
#Exploring the dataset
head(df)
#prints row and columns
dim(df)    
# Checking column names and their data types
column_types <- sapply(df, class)
print(column_types)
# Summary of the dataset
summary(df)
# Check for missing values
missing_values <- sapply(df, function(x) sum(is.na(x)))
missing_values
#Checking for any duplicate rows
duplicate_rows <- df[duplicated(df), ]
#prints the numbers of duplicate rows
print(duplicate_rows)
# Count the number of duplicate rows
num_duplicates <- nrow(duplicate_rows)
cat("Number of duplicate rows: ", num_duplicates, "\n")

# Checking outliers
# Z-score function
z_score <- function(x) {
  (x - mean(x)) / sd(x)
}

# Calculate Z-scores
z_scores_price <- z_score(df$Price)
z_scores_area <- z_score(df$Area)

# Identify outliers
outlier_indices_price <- which(abs(z_scores_price) > 3)
outlier_indices_area <- which(abs(z_scores_area) > 3)

# Show the outliers before removal
outliers_price <- df$Price[outlier_indices_price]
outliers_area <- df$Area[outlier_indices_area]
outliers_price
outliers_area

# Remove outliers
df_clean <- df[-c(outlier_indices_price, outlier_indices_area), ]

#shows the rows and columns after removing outliers
dim(df_clean)

# Show the cleaned dataframe
head(df_clean)



# 2.	Data Transformation:
# Standardizing 'Price', 'Area', and 'No..of.Bedrooms'
df_clean$Price <- scale(df_clean$Price)
df_clean$Area <- scale(df_clean$Area)
df_clean$No..of.Bedrooms <- scale(df_clean$No..of.Bedrooms)
summary(df_clean)

#3.	Encoding Categorical Variables

# Convert 'City' and 'Location' to factors
df_clean$City <- as.factor(df_clean$City)
df_clean$Location <- as.factor(df_clean$Location)

# Check data structure
str(df_clean)

#4. Data Reduction

# Remove the 'X' column
df_clean <- df_clean %>% select(-X)

#Shows rows and columns after removing 'X' column
dim(df_clean)


#Exploratory data analysis


#UNIVARIATE ANALYSIS

#1.  Histogram for Price
hist(df_clean$Price, main="Histogram of Prices", xlab="Price", col="blue")



#2. Histogram for Area
hist(df_clean$Area, breaks = 10, col = "lightgreen", xlab = "Area", main = "Histogram of Area")

#3. Histogram for No..of.Bedrooms
hist(df_clean$No..of.Bedrooms, breaks = 5, col = "orange", xlab = "Number of Bedrooms", main = "Histogram of Number of Bedrooms")

# 4.  Boxplot for Area
boxplot(df_clean$Area, main="Boxplot of Area", ylab="Area")

# 5. Density plot for No. of Bedrooms
plot(density(df_clean$No..of.Bedrooms), main="Density Plot of No. of Bedrooms", xlab="No. of Bedrooms")

# 6 Create bar plot of Distribution of cities
barplot(city_counts, col = "skyblue", main = "Distribution of Cities", xlab = "City", ylab = "Count")

#BIVARIATE ANALYSIS

#1. Scatter plot of Price vs Area

plot(df_clean$Price, df_clean$Area, pch = 16, col = "blue", xlab = "Price", ylab = "Area", main = "Scatterplot of Price vs. Area")


#2, Scatter plot of Price vs No. of Bedrooms
plot(df_clean$Price, df_clean$No..of.Bedrooms, main="Price vs No. of Bedrooms", xlab="Price", ylab="No. of Bedrooms", pch=19, col="darkred")

#3. visualizing Correlation matrix
cor_matrix <- cor(df_clean[, c("Price", "Area", "No..of.Bedrooms")])
print(cor_matrix)
corrplot(cor_matrix, method = "color", type = "lower", addCoef.col = "black", tl.col = "black", tl.srt = 45)


# Calculate correlation by group of city classes with Price
data = df_clean
correlation_by_city <- data %>%
  group_by(City) %>%
  summarize(correlation = cor(Price, No..of.Bedrooms))

# Print results
print(correlation_by_city)

#Visualizing correlation by  city with price

ggplot(correlation_by_city, aes(x = City, y = correlation)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  labs(title = "Correlation between Price and No. of Bedrooms by City",
       x = "City",
       y = "Correlation Coefficient")


#4 Correlation of categorical columns(City and Location) using ANNOVA

# Perform ANOVA for Location
anova_location <- aov(Price ~ Location, data = df_clean)
summary(anova_location)

# Perform ANOVA for City
anova_city <- aov(Price ~ City, data = df_clean)
summary(anova_city)

#5.  Boxplot of Price by Location
boxplot(Price ~ Location, data = df_clean, main="Price by Location", xlab="Location", ylab="Price", las=2, col="lightblue")

#6.# Boxplot of Area by Location
boxplot(Area ~ Location, data = df_clean, col = "lightgreen", xlab = "Location", ylab = "Area", main = "Boxplot of Area by Location")

#Building a Machine learning statistical model

#1.Applying Linear regression

set.seed(123)  # For reproducibility

# Splitting data into training (80%) and testing (20%) sets
train_index <- sample(seq_len(nrow(df_clean)), nrow(df_clean) * 0.8)
train_data <- df_clean[train_index, ]
test_data <- df_clean[-train_index, ]

# Fit linear regression model using only "No..of Bedroom" and "area" as independent variables
lm_model <- lm(Price ~ No..of.Bedrooms + Area, data = train_data)

# Summary of the model
summary(lm_model)

# Predictions on test set
predictions <- predict(lm_model, newdata = test_data)

# Evaluation of the model 
#Root Mean Square Error(RMSE)
rmse <- sqrt(mean((predictions - test_data$Price)^2))
rmse


# Mean Absolute Error (MAE)
mae <- mean(abs(predictions - test_data$Price))

# Mean Absolute Percentage Error (MAPE)
mape <- mean(abs((predictions - test_data$Price) / test_data$Price)) * 100

# R-squared (R²)
sse <- sum((predictions - test_data$Price)^2)
sst <- sum((test_data$Price - mean(test_data$Price))^2)
r_squared <- 1 - (sse / sst)

# Prints the result of evaluation metrics(Linear Regression)
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")
cat("R-squared:", r_squared, "\n")


# Plotting actual vs predicted prices
ggplot(test_data, aes(x = Price, y = predictions)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Prices(LinearRegression)", x = "Actual Price",y = "Predicted Price")


#2.Applying XGBoost Algorithm

# Splitting data into training (80%) and testing (20%) sets
set.seed(123)  # For reproducibility
train_index <- sample(seq_len(nrow(df_clean)), nrow(df_clean) * 0.8)
train_data <- df_clean[train_index, ]
test_data <- df_clean[-train_index, ]

# Prepares the data for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, c("No..of.Bedrooms", "Area")]), label = train_data$Price)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, c("No..of.Bedrooms", "Area")]))

# Fit the XGBoost model
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
xgb_model <- xgb.train(params = xgb_params, data = train_matrix, nrounds = 100)

# Predictions on test set
predictions <- predict(xgb_model, newdata = test_matrix)

# Evaluate the model 
rmse <- sqrt(mean((predictions - test_data$Price)^2))
mae <- mean(abs(predictions - test_data$Price))
mape <- mean(abs((predictions - test_data$Price) / test_data$Price)) * 100

sse <- sum((predictions - test_data$Price)^2)
sst <- sum((test_data$Price - mean(test_data$Price))^2)
r_squared <- 1 - (sse / sst)

# Print evaluation metrics(XGBoost)
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")
cat("R-squared:", r_squared, "\n")

# Plotting actual vs predicted prices

ggplot(test_data, aes(x = Price, y = predictions)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Prices (XGBoost)", x = "Actual Price", y = "Predicted Price")



#3.Applying Random Forest Classifier

# Splitting data into training (80%) and testing (20%) sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(df_clean$Price, p = 0.8, list = FALSE)
train_data <- df_clean[train_index, ]
test_data <- df_clean[-train_index, ]

# Fit a Random Forest model
rf_model <- randomForest(Price ~ No..of.Bedrooms + Area, data = train_data, ntree = 100)

# Predict on the test set
rf_predictions <- predict(rf_model, newdata = test_data)

# Evaluate the model
rf_rmse <- sqrt(mean((rf_predictions - test_data$Price)^2))
rf_mae <- mean(abs(rf_predictions - test_data$Price))
rf_mape <- mean(abs((rf_predictions - test_data$Price) / test_data$Price)) * 100

sse <- sum((rf_predictions - test_data$Price)^2)
sst <- sum((test_data$Price - mean(test_data$Price))^2)
rf_r_squared <- 1 - (sse / sst)

# Prints the result of evaluation metrics(RandomForest)
cat("RMSE:", rf_rmse, "\n")
cat("MAE:", rf_mae, "\n")
cat("MAPE:", rf_mape, "%\n")
cat("R-squared:", rf_r_squared, "\n")

# Plotting actual vs predicted prices
ggplot(test_data, aes(x = Price, y = rf_predictions)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Prices (Random Forest)", x = "Actual Price", y = "Predicted Price")


#Comparing the different model based on the obtained values of evaluation metrics


# Data (scaled MAPE values for better visualization)

models <- c("Linear Regression", "XGBoost", "Random Forest Classifier")
RMSE <- c(0.994, 1.00, 0.935)
MAE <- c(0.61, 0.597, 0.591)
MAPE <- c(0.172, 0.168, 0.148)  # Scaled down MAPE values (divided by 1000)
R_squared <- c(0.077, 0.062, 0.086)

# Create data frame
evaluation_data <- data.frame(models, RMSE, MAE, MAPE, R_squared)

# Reshape data for plotting
evaluation_data_long <- pivot_longer(evaluation_data,
                                     cols = c(RMSE, MAE, MAPE, R_squared),
                                     names_to = "Metric",
                                     values_to = "Value")

# Plotting
ggplot(evaluation_data_long, aes(x = Metric, y = Value, fill = models)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Comparison Based on Evaluation Metrics",
       x = "Metrics", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("Linear Regression" = "#1f77b4", 
                               "XGBoost" = "#ff7f0e", 
                               "Random Forest Classifier" = "#2ca02c"))
