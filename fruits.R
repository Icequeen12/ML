library(keras)
library(tensorflow)

fruit_list <- c("Apricot","Avocado","Banana","Blueberry","Cauliflower","Clementine","Cocos","Eggplant","Ginger","Grapefruit","Kiwi","Lemon","Limes","Lychee","Mandarine","Nectarine","Onion","Orange","Peach","Pear","Pepper","Pineapple","Plum","Pomegranate","Potato","Raspberry","Strawberry","Tomato")

target_size <- c(20,20)

train_image_files_path <- "C:/Users/Admin/Desktop/baza/Trening/"
valid_image_files_path <- "C:/Users/Admin/Desktop/baza/Test/"

train_data_gen = image_data_generator(
  rescale = 1/255
)

valid_data_gen <- image_data_generator(
  rescale = 1/255
)

train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    seed = 42)

valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    seed = 42)


train_samples <- train_image_array_gen$n
valid_samples <- valid_image_array_gen$n

batch_size <- 32
epochs <- 10

model <- keras_model_sequential()


model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'same', input_shape = c(img_width, img_height, 3)) %>%
  layer_activation('relu') %>%
  
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = 'same') %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation('relu') %>%
  layer_dropout(0.5) %>%
  
  layer_dense(output_n) %>%
  layer_activation('softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = 'accuracy'
)

hist <- model %>% fit_generator(
  train_image_array_gen,
  
  steps_per_epoch = as.integer(train_samples / batch_size),
  epochs = epochs,
  
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  verbose = 2,
  callbacks = list(
    callback_model_checkpoint("C:/Users/Admin/Desktop/baza/fruits_checkpoints.h5", save_best_only = TRUE)
  )
)

