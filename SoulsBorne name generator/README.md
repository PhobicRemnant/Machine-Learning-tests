#Soulsborne name generator

I've always been a fan of From Software games but I think that somehow it delves into the trope of dark fantasy names,
a trope I am completely onboard with. In this proyect the solutions are based on RNNs to see if with some help from the 
network we can build names that resemble the originals.

# Arquitecture_v001:

The initial arquitecture is based off the main tutorial of the TensorFlow page but this model uses far less data and double
GRU layers. The initial data set is conformed by only the main bosses from the SoulsBorne series, no Sekiro names or no character names. 
The results are quite, miniscule, the model simply imitates some words based on its own. 

## Notes on v001

Please note that the model is overfitted due to the small size of the learning set, it is unadvised to use a supervised learning model without dividing the dataset into a train and test populations. The good news is that there are more names to use in the SoulsBorne series, more character names will make a beter job at creating new names. If you try the model in the v001 version you can see that the names are strangely embedded (due to overfitting) into a sequence. 

Therefore, v002 will expand the dataset to be able to create strange names. Please be advised that the current nature of the project is not withing the boundaries of NPL.
