import pickle
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20, 8)

def plot_total_loss(hist_fname, model_name):
    """
    :param hist_fname: history file which contains the evolution of all the learning metrics
    :param model_name: model name
    :return: plot and save the evolution of the Elbo loss function
    """
    
    with open(hist_fname,'rb') as f:
        h = pickle.load(f)

    plt.plot(h['loss'],label='loss')
    plt.plot(h['val_total_loss'],label='val_loss')
    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.xlim(0,len(h['loss']))
    
    plt.title('Loss and Val_Loss of ' + model_name +' training')
    plt.legend()

    plt.savefig('training_history/'+model_name+'_loss.svg',format='svg', dpi=1200)
    plt.close()

def plot_kl_loss(hist_fname, model_name):
    """
    :param hist_fname: history file which contains the evolution of all the learning metrics
    :param model_name: model name
    :return: plot and save the evolution of the KL-divergence loss function
    """
    
    with open(hist_fname,'rb') as f:
        h = pickle.load(f)

    plt.plot(h['kl_loss'],label='kl_loss')
    plt.plot(h['val_kl_loss'],label='val_kl_loss')
    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.xlim(0,len(h['kl_loss']))
    
    plt.title('Loss and Val_Loss of ' + model_name +' training')
    plt.legend()

    plt.savefig('training_history/'+model_name+'_klloss.svg',format='svg', dpi=1200)
    plt.close()

def plot_reconstruction_loss(hist_fname, model_name):
    """
    :param hist_fname: history file which contains the evolution of all the learning metrics
    :param model_name: model name
    :return: plot and save the evolution of the reconstruction loss function
    """
    
    with open(hist_fname,'rb') as f:
        h = pickle.load(f)

    plt.plot(h['reconstruction_loss'],label='reconstruction_loss')
    plt.plot(h['val_reconstruction_loss'],label='val_reconstruction_loss')
    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.xlim(0, len(h['reconstruction_loss']))
    
    plt.title('Loss and Val_Loss of ' + model_name +' training')
    plt.legend()

    plt.savefig('training_history/'+model_name+'_reconstloss.svg',format='svg', dpi=1200)
    plt.close()