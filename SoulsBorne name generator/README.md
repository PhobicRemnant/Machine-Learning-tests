#Soulsborne name generator

I've always been a fan of From Software games but I think that somehow it delves into the trope of dark fantasy names,
a trope I am completely onboard with. In this proyect the solutions are based on RNNs to see if with some help from the 
network we can build names that resemble the originals.

# Arquitecture_v001:

The initial arquitecture is based off the main tutorial of the TensorFlow page but this model uses far less data and double
GRU layers. The initial data set is conformed by only the main bosses from the SoulsBorne series, no Sekiro names or no character names. 
The results are quite, miniscule, the model simply imitates some words based on its own. 

