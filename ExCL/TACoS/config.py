
class DefaultConfig(object):
    
    video_fts_path = 'D:\\Data\\Text-to-Clip\\SCDM\\data\\TACOS\\tacos_c3d_fc6_nonoverlap.hdf5'
    video_list = 'D:\\Data\\Text-to-Clip\\SCDM\\data\\TACOS\\datasplit_info\\tacos_split.npz'
    wordtoix_path = 'D:\\Data\\Text-to-Clip\\SCDM\\grounding\\TACOS\\words\\wordtoix.npy'
    word_fts_path = 'D:\\Data\\Text-to-Clip\\SCDM\\grounding\\TACOS\\words\\word_glove_fts_init.npy'
    train_data_path = 'D:\\Data\\Text-to-Clip\\SCDM\\data\\TACOS\\datasplit_info\\train.json'
    val_data_path = 'D:\\Data\\Text-to-Clip\\SCDM\\data\\TACOS\\datasplit_info\\val.json'
    test_data_path = 'D:\\Data\\Text-to-Clip\\SCDM\\data\\TACOS\\datasplit_info\\test.json'

    model_save_path = 'D:\\Data\\Text-to-Clip\\ExCL\\TACoS\\ckpt\\'

    vLSTM_embedding_dim = 4096 # C3D feature input
    vLSTM_hidden_dim = 256 
    tLSTM_embedding_dim = 300
    tLSTM_hidden_dim = 256 
    sLSTM_embedding_dim = 256*2+256*2 # vLSTM_hidden_dim*2+tLSTM_hidden_dim*2
    sLSTM_hidden_dim = 128
    MLP_input_dim =  256*2+256*2+128*2
    MLP_hidden_dim = 256
    output_dim = 1024 # ANet\TACoS output

    number_layers = 1
    dropout = 0.5 # value in paper
    lr = 0.001 # value in paper
    batch_size = 32 # value in paper
    train_epochs = 30 # value in paper
    start_epoch = 0

    
