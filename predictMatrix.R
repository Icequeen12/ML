library(keras)
library(tensorflow)

model <- load_model_hdf5("C:/Users/Admin/Desktop/baza/fruits_checkpoints.h5")

img_path <- "C:/Users/Admin/Desktop/testownik/"

valid_data_gen <- image_data_generator(
  rescale = 1/255
)

target_size <- c(20,20)

valid_image_array_gen <- flow_images_from_directory(img_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    seed = 42)

valid_image_array_gen$n

#steps = as.integer(valid_samples / )

pred <- model %>% predict_generator(valid_image_array_gen,1)
pred <- format(round(pred, 2), nsamll = 4)

write.table(pred, file = "matrixFruits.txt", sep = "\t", row.names = FALSE)
