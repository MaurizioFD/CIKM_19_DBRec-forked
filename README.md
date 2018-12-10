# DBRec
Dual-Bridging Recommendation with Latent Group Discovery

's_matrix.npy' size of [num_user, num_user] is the user similar matrix, which can be obtain from the rating matrix. 

The dataset is Amozon music data, and it is split into training set(70%), development set (10%), and testing set (20%)

Each ground truth item is ranked along with 99 randomly sampled item, and HR@k and ndcg@k is measured, where k is in the range of [1,10]

Pretrain the network to get user and item embedding matrix, and then employ clustering to dual-bridge similar items/users in a group, so that we can capture user preferences and item characteristics at different level of granularities. 

tensorflow, keras, pickle, numpy are required in order to run the code.

To train the model, type in the commend:

python3 model_softmax.py train 35

where 35 is the pre-specified latent group number, and it can be fine tuned to get the best performance. 

To test the model, type in the commend:

python3 model_softmax.py test 35
