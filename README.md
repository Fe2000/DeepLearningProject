# DeepLearningProject
Image Captioning is the process of generating a textual description of an image. It uses both Natural  Language Processing and Computer Vision to generate the captions. The work that I did falls in  that category and uses both text and vision techniques to accomplish the task at hand.

# Background

To cover some background over this topic one paper I look over was Show and Tell: A Neural Image
 Caption Generator: In this paper, They presented a generative model based on a deep recurrent
 architecture that combines recent advances in computer vision and machine translation and that
 can be used to generate natural sentences describing an image (cited over 5000 times)

 
# Specification

 The main techniques that are the core of my implementation were transfer learning, neural networks,
 and word embeddings. Transfer learning is a machine learning technique that focuses on storing
 knowledge gained while solving one problem and applying it to a different but related problem. A
 neural network is a series of algorithms that endeavors to recognize underlying relationships in a set
 of data through a process that mimics the way the human brain operates. Word embeddings are a
 dense representation of a word. The data that I used for this project was called the Flickr 8k dataset
 which consisted of 8,000 images that are each paired with five different captions which provide clear
 descriptions

<img width="1800" height="378" alt="image" src="https://github.com/user-attachments/assets/416bcdcd-8ced-4a5f-9f5d-61c96b3e02a2" />


 # Implementation


  The main algorithm that was able to caption a image consists of using deep learning techniques
 such as Long short-term memory (LSTM) neural network model. Some libraries and tools I used
 for the project were Tensorflow and keras to build the model, pickle to save and read in pickle files,
 cv2 and numpy to open and manipulate images. The dataset was comprised of images with there
 appropriate caption. To pre-process the text part of the data I use basic NLP pre-processing like
turning everything into lowercase and removing any punctuation. Next I created word embedding
 vectors for each unique word for a fixed length. I used pre-trained glove embeddings. To handle
 the image part of the data I used technique called transfer learning, which uses an already trained
 model to transform the original image to a vector. The pre-trained model I used was InceptionV3 (did not upload this model, you can find this on Kaggle or other sites)
 and I removed the last layer so I could make the output be a set vector size of 2048. Once all the
 pre-processing steps were done, I made a model which had dropout, embedding and LSTM layers to
 predict the caption of the given image.

<img width="1039" height="526" alt="image" src="https://github.com/user-attachments/assets/01ce6c55-b39c-4dfd-9132-13409381e31a" />


 # Evaluation
 Once the model was made I trained on ten epochs. Each epoch took six minutes and total train time
 took around an hour. The first loss value was around 524 and once the model was done training
 the loss lowered to 25.

 
 # Conclusions
 This project showed that image captioning is possible using deep learning techniques. The significance
 of this project is high since the task of automatically producing a natural-language description for
 an image, has the potential to assist those with visual impairments by explaining images using
 text-to-speech systems.
