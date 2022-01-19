from timeit import repeat
from pathlib import Path
import pickle as pk
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


FOLDER = Path("/home/pierre/Data/VBDM_data/")
DATASET_ORIGINAL = Path("population_dataset_step.50.pk")
DATASET_SUBSAMPLED = Path("population_dataset_step.200.pk")

# if __name__ == '__main__':

#     ###############################################################################
#     # Original version
#     datafile = DATASET_ORIGINAL
#     with open(FOLDER/datafile, 'rb') as f:
#         X, y, bins = pk.load(f)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=12345)

#     proc_time_o = repeat(stmt='cross_temporal_decoding(X_train, X_test, y_train, y_test)',
#                        setup='from cross_temporal_decoding import cross_temporal_decoding',
#                        globals={'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
#                        number=1, repeat=5)

#     print('Original algorithm')
#     print(np.mean(proc_time_o), np.std(proc_time_o))

###############################################################################
# Downsampled version
datafile = DATASET_SUBSAMPLED
with open(FOLDER/datafile, 'rb') as f:
    X, y, bins = pk.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=12345)

    # proc_time_ds = repeat(stmt='cross_temporal_decoding(X_train, X_test, y_train, y_test)',
    #                    setup='from cross_temporal_decoding import cross_temporal_decoding',
    #                    globals={'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
    #                    number=1, repeat=10)

    # print('Downsampled dataset')
    # print(np.mean(proc_time_ds), np.std(proc_time_ds))

    # ###############################################################################
    # # Non repeated training
    # proc_time_nrt = repeat(stmt='CTD_non_repeated_training(X_train, X_test, y_train, y_test)',
    #                    setup='from cross_temporal_decoding import CTD_non_repeated_training',
    #                    globals={'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
    #                    number=1, repeat=50)

    # print('Non-repeated training')
    # print(np.mean(proc_time_nrt), np.std(proc_time_nrt))

    # ###############################################################################
    # # Vectorized testing
    # proc_time_vectest = repeat(stmt='CTD_vectorized_testing(X_train, X_test, y_train, y_test)',
    #                            setup='from cross_temporal_decoding import CTD_vectorized_testing',
    #                            globals={'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
    #                            number=1, repeat=50)

    # print('Vectorized testing')
    # print(np.mean(proc_time_vectest), np.std(proc_time_vectest))

###############################################################################
# Vectorized training
proc_time_vectrain = repeat(stmt='vectorized_CTD(X_train, X_test, y_train, y_test)',
                            setup='from cross_temporal_decoding import vectorized_CTD',
                            globals={'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
                            number=1, repeat=50)

print('Vectorized training')
print(np.mean(proc_time_vectrain), np.std(proc_time_vectrain))

    # ###############################################################################
    # # Plotting the processing time with each version of the algorithm
    # timings = [proc_time_o, proc_time_ds, proc_time_nrt, proc_time_vectest, proc_time_vectrain]
    # means = np.array([np.mean(elt)*1000 for elt in timings])
    # algos = ['original', 'down-sampled', 'single training', 'vectorized testing', 'vectorized training']

    # plt.ion()
    # sns.set(font_scale=1.2)

    # plt.figure(figsize=(5, 5))
    # plt.plot(algos, means, marker='x')
    # plt.xticks(rotation=45)
    # plt.yscale('log')
    # plt.ylabel('processing time (ms)')
    # plt.tight_layout()
    # plt.show()
