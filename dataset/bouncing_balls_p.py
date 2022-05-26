from .dataset_base import BaseDataset
from torch.utils.data import DataLoader
import numpy as np
import os

color_list=[(1.,1.,1.), #white
            (1.,0.,0.), #red
            (0.,1.,0.), #green
            (0.,0.,1.), #blue
            (1.,1.,0.), #yellow
            (1.,0.,1.), #magenta
            (0.,1.,1.)] #cyan

class ColoredBouncingBallsStackedOnlinegen(BaseDataset):
    def __init__(self, directory, yp_window_len, y_window_len, x_window_len, window_stride, 
                       num_seq_dataset=1000, rgb_label=False, seed=0, switch_samples_per_sequence=200, switch_probability = .2, 
                       patch_loc=(1,1), baseline_train=False, label_noise=0., random_loc=False, num_distractor=0, random_order=True,
                       no_patch=False, split_name='train', **kwargs):
        super().__init__()
        all_Y = np.load(os.path.join(directory, 'images_data.npy'),mmap_mode='r')
        all_X = np.load(os.path.join(directory, 'balls_color.npy'),mmap_mode='r')

        num_ic, num_class, seq_len, ch, res_square = all_Y.shape
        num_seq = num_ic * num_class # per initial condition, seqs for all colors are stacked 

        self.seed = seed
        self.y_window_len = y_window_len
        self.yp_window_len = yp_window_len
        if not (self.yp_window_len == 1):
            raise NotImplementedError()
        self.window_stride = window_stride
        self.switch_samples_per_sequence = switch_samples_per_sequence # wasn't used?
        self.switch_probability = switch_probability
        self.no_patch = no_patch
        self.default_num_seq = num_seq_dataset

        #hard code to only take the first num_seq_dataset sequences for memory. TODO make the dataset in memory friendly format
        #resulting number of sequences in the returned dataset is (seq_len - yp_window_len - y_window_len) // window_stride + 1
        # print(num_seq)
        num_seq=min(num_seq,num_seq_dataset)
        num_ic = num_seq // num_class
        self.all_Y = all_Y[:num_ic]
        self.all_X = all_X[:num_ic]

        res = int(np.sqrt(int(res_square)))
        self.res = res
        self.all_Y = self.all_Y.reshape(num_ic, num_class, seq_len, ch, res, res) # need the extra ic dimension to ensure permutation is done among the same ball trajs
        assert (self.all_X.shape[0] == num_ic and self.all_X.shape[1] == num_class) and self.all_X.shape[2] == seq_len

        #arranging the sequences
        sample_len = y_window_len + yp_window_len
        cnt_in_seq = (seq_len - sample_len) // window_stride + 1
        self.sample_len = sample_len
        self.cnt_in_seq = cnt_in_seq

        self.num_ic = num_ic
        self.num_class = num_class
        
        self.random_loc = random_loc
        if self.random_loc:
            self.patch_loc_x = None
            self.patch_loc_y  =None
        else:
            self.patch_loc_x, self.patch_loc_y = patch_loc
        self.baseline_train = baseline_train
        self.label_noise = label_noise
        self.num_distractor = num_distractor
        self.random_order = random_order
        self.split_name = split_name

        if rgb_label == True:
            self.int_to_rgb()

    def __getitem__(self, idx):
        # an index has to enumerate: which initial condition, which color, which strided window in that sequence, and which sample for the last two (switch_samples_per_sequence)
        ic = idx % self.num_ic
        clf = idx % self.num_class
        stridepos = idx % self.cnt_in_seq
        # samplenum = idx % self.switch_samples_per_sequence

        startseq = stridepos*self.window_stride
        endseq = stridepos*self.window_stride + self.y_window_len
        endseq_p = stridepos*self.window_stride + self.sample_len

        window_img = np.copy(self.all_Y[ic, clf, startseq: endseq_p])
        window_lbl = np.copy(self.all_X[ic, clf, startseq: endseq_p])
        window_lbl_o = np.copy(self.all_X[ic, clf, startseq: endseq_p])
        window_img_p = np.copy(self.all_Y[ic, clf, startseq: endseq_p]) # label is image with one pixel perturbation
        
        # need to create a local random generator so as to not change the global seed
        if self.split_name == 'test':
            possible_length = int(2*(self.default_num_seq*self.cnt_in_seq*self.switch_samples_per_sequence))
            generator = np.random.default_rng(seed=(idx+possible_length))
        elif self.split_name == 'valid':
            possible_length = int(self.default_num_seq*self.cnt_in_seq*self.switch_samples_per_sequence)
            generator = np.random.default_rng(seed=(idx+possible_length))
        else:
            generator = np.random.default_rng(seed=idx)
        if generator.random() <= self.switch_probability:
            #switches this balls with another in the same initial condition
            switched_class_idx = generator.integers(0,self.num_class)
            window_img[self.y_window_len : self.sample_len] = self.all_Y[ic, switched_class_idx, endseq:endseq_p]
            window_lbl[self.y_window_len - 1 : self.sample_len] = self.all_X[ic, switched_class_idx, endseq-1:endseq_p]
            window_img_p[self.y_window_len : self.sample_len] = self.all_Y[ic, switched_class_idx, endseq:endseq_p]
        
        if not (self.no_patch):
            if not (self.random_loc):
                pl_x = self.patch_loc_x
                pl_y = self.patch_loc_y
            else:
                loc_seed = generator.random()
                if loc_seed <= 0.25:
                    pl_x = 1
                    pl_y = generator.integers(1, int(self.res)-1)
                elif loc_seed <= 0.5:
                    pl_y = 1
                    pl_x = generator.integers(1, int(self.res)-1)
                elif loc_seed <= 0.75:
                    pl_y = int(self.res)-2
                    pl_x = generator.integers(1, int(self.res)-1)
                else:
                    pl_y = int(self.res)-2
                    pl_x = generator.integers(1, int(self.res)-1)
            window_img = np.tile(window_img, (1, int(self.num_distractor), 1, 1))
            window_img_p = np.tile(window_img_p, (1, int(self.num_distractor)+1, 1, 1))
            if self.random_order:
                true_pixel_idx = generator.integers(0,int(self.num_distractor)+1)
            else:
                true_pixel_idx = 0

            for n_s in range(0, true_pixel_idx):
                for t in range(self.sample_len-self.yp_window_len):
                    r_cls = generator.integers(0,self.num_class)
                    for c in range(len(color_list[0])):
                        img_d_idx= n_s*len(color_list[0])+c
                        img_p_d_idx = n_s*len(color_list[0])+c
                        window_img[t,img_d_idx, pl_x, pl_y] = color_list[int(r_cls)][c]
                        window_img_p[t,img_p_d_idx, pl_x, pl_y] = color_list[int(r_cls)][c]
                        if self.label_noise > 0.:
                            if color_list[int(r_cls)][c] > 0.5:
                                r = generator.random()
                                window_img[t, img_d_idx, pl_x, pl_y] -= r * self.label_noise
                                window_img_p[t, img_p_d_idx, pl_x, pl_y] -= r * self.label_noise
                            elif color_list[int(r_cls)][c] < 0.5:
                                r = generator.random()
                                window_img[t, img_d_idx, pl_x, pl_y] += r * self.label_noise
                                window_img_p[t, img_p_d_idx, pl_x, pl_y] += r * self.label_noise
            for n_s in range(true_pixel_idx+1, int(self.num_distractor)+1):
                for t in range(self.sample_len-self.yp_window_len):
                    r_cls = generator.integers(0,self.num_class)
                    for c in range(len(color_list[0])):
                        img_d_idx= (n_s-1)*len(color_list[0])+c
                        img_p_d_idx = n_s*len(color_list[0])+c
                        window_img[t,img_d_idx, pl_x, pl_y] = color_list[int(r_cls)][c]
                        window_img_p[t,img_p_d_idx, pl_x, pl_y] = color_list[int(r_cls)][c]
                        if self.label_noise > 0.:
                            if color_list[int(r_cls)][c] > 0.5:
                                r = generator.random()
                                window_img[t, img_d_idx, pl_x, pl_y] -= r * self.label_noise
                                window_img_p[t, img_p_d_idx, pl_x, pl_y] -= r * self.label_noise
                            elif color_list[int(r_cls)][c] < 0.5:
                                r = generator.random()
                                window_img[t, img_d_idx, pl_x, pl_y] += r * self.label_noise
                                window_img_p[t, img_p_d_idx, pl_x, pl_y] += r * self.label_noise

            for n_s in range(true_pixel_idx, true_pixel_idx+1):
                for t in range(self.sample_len-self.yp_window_len):
                    for c in range(len(color_list[0])):
                        img_p_d_idx = n_s*len(color_list[0])+c
                        window_img_p[t,img_p_d_idx, pl_x, pl_y] = color_list[int(window_lbl[t])][c]
                        if self.label_noise > 0.:
                            if color_list[int(window_lbl[t])][c] > 0.5:
                                window_img_p[t, img_p_d_idx, pl_x, pl_y] -= generator.random() * self.label_noise
                            elif color_list[int(window_lbl[t])][c] < 0.5:
                                window_img_p[t, img_p_d_idx, pl_x, pl_y] += generator.random() * self.label_noise


        if self.baseline_train:
            return window_img[self.y_window_len : self.sample_len, :len(color_list[0])], window_img_p[:self.y_window_len], window_lbl
        else:
            return window_img[self.y_window_len : self.sample_len, :len(color_list[0])], window_img[:self.y_window_len], window_img_p

    def __len__(self):
        # an index has to enumerate: which initial condition, which color, which strided window in that sequence, and which sample for the last two (switch_samples_per_sequence)
        # therefore length is num_ic*num_class*cnt_in_seq*switch_samples_per_sequence

        return self.num_ic*self.num_class*self.cnt_in_seq*self.switch_samples_per_sequence

    def int_to_rgb(self):
        # add an extra dimension recording rgb values
        dim_0, dim_1, dim_2 = self.all_X.shape
        tmp1 = np.expand_dims(self.all_X, axis=3)
        tmp = np.repeat(tmp1, 3, axis=3)
        for i in range(dim_0):
            for j in range(dim_1):
                for m in range(dim_2):
                    c_idx = int(tmp1[i,j,m,0])
                    for k in range( len(color_list[0]) ):
                        assert tmp[i,j,m,k] == tmp1[i,j,m,0]
                        tmp[i,j,m,k] = color_list[c_idx][k]
        self.all_X = tmp