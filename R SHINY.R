library(shiny)
library(ggplot2)

ui <- fluidPage(
  titlePanel("5 Plots from mtcars"),
  fluidRow(column(6,plotOutput("p1")),column(6,plotOutput("p2"))),
  fluidRow(column(6,plotOutput("p3")),column(6,plotOutput("p4"))),
  fluidRow(column(12,plotOutput("p5")))
)

server <- function(input, output) {
  output$p1 <- renderPlot(ggplot(mtcars,aes(mpg)) + geom_histogram(bins=10,fill="skyblue") + 
                            ggtitle("1. MPG Distribution"))
  output$p2 <- renderPlot(ggplot(mtcars,aes(wt,mpg)) + geom_point(color="tomato") + 
                            ggtitle("2. Weight vs MPG"))
  output$p3 <- renderPlot(ggplot(mtcars,aes(factor(cyl),hp)) + geom_boxplot(fill="lightgreen") + 
                            ggtitle("3. HP by Cylinders"))
  output$p4 <- renderPlot(ggplot(mtcars,aes(factor(gear))) + geom_bar(fill="gold") + 
                            ggtitle("4. Gear Count"))
  output$p5 <- renderPlot(ggplot(mtcars,aes(qsec,hp)) + geom_line(color="navy") + 
                            ggtitle("5. HP vs Quarter Mile"))
}

shinyApp(ui, server)