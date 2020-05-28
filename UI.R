library(shiny)
library(keras)

model <- load_model_hdf5("C:/Users/Admin/Desktop/baza/fruits_checkpoints.h5")

shinyUI(fluidPage(
  
  titlePanel("Klasyfikacja owoców"),
  
  sidebarLayout(
    sidebarPanel(
      textInput("path", "Ścieżka:",),
      actionButton("goButton", "Analizuj")
    ),
    
    mainPanel(
      h3(textOutput("path", container = span)),
      textOutput(outputId = "prediction"),
      plotOutput(outputId = "image")
    ),
  )
)
)