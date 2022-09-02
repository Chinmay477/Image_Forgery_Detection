import torch
from src.cnn.cnn import CNN
from src.feature_fusion.feature_vector_generation import create_feature_vectors


with torch.no_grad():
    model = CNN()
    model.load_state_dict(torch.load('../src/Cnn.pt',
                                     map_location=lambda storage, loc: storage))
    model.eval()
    model = model.double()

    authentic_path = '../data/CASIA2/Au/*'
    tampered_path = '../data/CASIA2/Tp/*'
    output_filename = 'features.csv'
    create_feature_vectors(model, tampered_path, authentic_path, output_filename)
