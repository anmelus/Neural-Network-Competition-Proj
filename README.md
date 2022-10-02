# Neural-Network-Competition-Proj

This project was for a neural networks course at North Carolina State University and was a competition project between ~32 teams of 3 people each. Unforunately, one of our group's members disappeared so it became a 2-man project.

## Background

There are about 2 million people in the United States alone living with at least one amputated limb with millions more outside the nation. Receiving an amputation is alone a life-changing experience and the loss of a leg completely changes how a person function on a day-to-day basis. For this reason, prosthetics are a common solution for amputees; however, they are still very far from being perfect solutions and frequently require amputees to require a cane and walk in an awkward manner to even get around. For example, a person has to consciously change the way they walk dependning on if they are walking upstairs, downstairs, on grass, or on pavement. The purpose of this project is to attempt to apply a neural network to detect the type of surface the user is walking on and in the future potentially prompt some mechanisms to assist with walking on different surfaces.

For this project, 6 individuals strapped an gyroscope and acceleromotor to their legs and walked around an area with stairs, grass, and pavement. Example data can be seen in the TrainingData folder. This data was used to train a neural network to recognize which type of ground the user was walking on.

## Methodology

A more in depth methodology as to how we solved this problem is viewable in the final report. Our team decided on using a convolutional neural network. The input layer is shaped in accordance with the sampling frequency. Additionally, a filter was used to prevent the model from predicting a fraction of a second of walking on the ground and then immediately predicting that the user was walking upstairs for a fraction of a second. More data on hyperparameters, model tuning, etc, are available in the project report.

## Conclusion

I worked with Radek Pudelko https://github.com/RadekPudelko on this project and we ended up getting 3rd place with an f1 score of about ~92% on the final test set. On our own validation sets we hit scores of ~95%.
