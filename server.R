shinyServer(function(input, output) {
  
  fruit_list <- c("morela","awokado", "banan", "jagoda", "kalafior", "klementynka", "kokos", "bakłażan", "imbir", "grejpfrut", "kiwi", "cytryna", "limonka", "liczi","mandarynka", "nektarynka", "cebula", "pomarańcza", "brzoskwinia", "gruszka", "papryka", "ananas", "śliwka", "granat", "ziemniak", "malina", "truskawka", "pomidor")
  
  plik <- eventReactive(input$goButton, {
    input$path
  })
  
  image <- reactive({
    jpeg::readJPEG(plik())
  })
  output$prediction <- renderText({
    
    img <- image_load(plik(), target_size = c(20,20))
    img_tensor <- image_to_array(img)
    img_tensor <- array_reshape(img_tensor, c(1, 20, 20, 3))
    img_tensor <- img_tensor / 255
    
    a<- model %>% predict_classes(img_tensor)
    paste("Sądzę żę to :  ", fruit_list[a+1])
  })
  
  output$image <- renderPlot({
    plot(as.raster(image()))}, height = 250, width = 250)
})