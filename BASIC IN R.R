# Load the mtcars dataset
data(mtcars)
head(mtcars)
# Plot a histogram of mpg
hist(mtcars$mpg, main = "Histogram of MPG", xlab = "Miles Per Gallon")



# Load the iris dataset
data(iris)
# Plot a scatter plot of Sepal Length
plot(iris$Sepal.Width, iris$Sepal.Length, main = "Scatter Plot of Sepal Length vs. Sepal Width", xlab = "Sepal Width", ylab = "Sepal Length")

# Load the airquality dataset
data(airquality)
# Plot a boxplot of Ozone levels by Month
boxplot(Ozone ~ Month, data = airquality, main = "Boxplo

