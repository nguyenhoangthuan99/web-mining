import pickle

with open('data/toy_dataset.pickle', 'rb') as handle:
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(handle)

print(ratings_list)

