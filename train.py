import torch
import torch.nn.functional as F
from torch import  optim
from itertools import  chain
import time
import numpy as np
import utils
import  dataset
import  models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_encoder_configuration():
    num_hidden_layers = 1
    hidden_sizes = [100]
    activations = [F.relu]
    return num_hidden_layers, hidden_sizes, activations

def set_decoder_configuration():
    hidden_dim = 50
    embedding_dim = 50
    return hidden_dim, embedding_dim

def set_learning_parameters():
    num_epoches = 100
    batch_size = 64
    learning_rate = 10**-3
    return num_epoches, batch_size, learning_rate

def topic_loss(encoder, decoder, doc_batch, tau, V, B):
    # print('***************************************')
    mus, log_sigmas = encoder(doc_batch,device)
    samples = utils.reparameterize(mus, log_sigmas)
    # print('\t\tmus.shape = ', mus.shape,' log_sigma.shape = ', log_sigmas.shape)
    thetas = samples.softmax(dim = 1)
    hidden_states, labels = decoder(doc_batch,device)
    print('\t\tthetas.shape = ', thetas.shape)
    print('\t\thidden_states.shape = ', hidden_states.shape)
    # print('\t\tlabels.shape = ', labels.shape)

    b = torch.matmul(hidden_states,tau)
    b = b.sigmoid()
    b = b.view(hidden_states.shape[0], hidden_states.shape[1],1)
    logits1 = torch.matmul(hidden_states, V)

    bias = torch.matmul(thetas,B)
    bias = bias.view(decoder.batch_size,1,decoder.num_words)
    print('\t\tlogits1.shape = ', logits1.shape)
    print('\t\tbias.shape = ', bias.shape)

    logits0 = logits1 + bias

    # print('\t\tb.shape = ', b.shape)
    print('\t\tlogits0.shape = ', logits0.shape)
    # print('\t\tlogits1.shape = ', logits1.shape)

    probs = logits1.softmax(dim=2)*b + (1-b)*logits0.softmax(dim=2)
    # assert (probs.cpu() < 0).sum() == 0, 'there must be something wrong'
    # probs += (1-b)*logits.softmax(dim=2)

    # print('\t\tprobs.shape = ', probs.shape)
    # print('\t\tprobs.sum = ', probs.sum().item())

    probs = probs.view(-1,decoder.num_words)
    labels = labels.view(-1)
    masks = (labels > 0).float()

    loss = -1*torch.sum(torch.log(probs[range(probs.shape[0]),labels])*masks)/torch.sum(masks).item()
    # print('after flatten: probs.shape = ', probs.shape)
    # print('labels.shape = ', labels.shape)
    # print('labels = ', labels)

    loss += utils.kld(mus, log_sigmas)
    return loss

def sparseTopicTransformer_loss(encoder, decoder, doc_batch):
    mus, log_sigmas = encoder(doc_batch,device)
    samples = utils.reparameterize(mus, log_sigmas)
    # print('\t\tmus.shape = ', mus.shape,' log_sigma.shape = ', log_sigmas.shape)
    thetas = samples.softmax(dim = 1)
    hidden_states, labels = decoder(doc_batch,device)
    print('\t\thidden_states.shape = ', hidden_states.shape)
    print('\t\tlabels.shape = ', labels.shape)
    print('***************************************')
    loss = utils.kld(mus, log_sigmas) + utils.recovery_loss(hidden_states, samples) + utils.sparsity_regularization(thetas)
    return loss

def main(data_file, num_topics):
    data = dataset.corpus(data_file)

    print('data: #docs = %d, #words = %d' %(data.num_docs,data.num_words))



    num_words = data.num_words
    num_hidden_layers, hidden_sizes, activations = set_encoder_configuration()
    hidden_dim, embedding_dim = set_decoder_configuration()
    num_epoches, batch_size, learning_rate = set_learning_parameters()

    encoder = models.MLPEncoder(num_words, num_topics, num_hidden_layers, hidden_sizes, activations, batch_size)
    encoder = encoder.to(device)
    decoder = models.topicTransformer(hidden_dim, embedding_dim, num_words, batch_size)
    decoder = decoder.to(device)

    tau = torch.randn((hidden_dim),requires_grad=True, device=device)
    print (hidden_dim)
    V = torch.randn((hidden_dim,num_words),requires_grad=True, device=device)
    B = torch.randn((num_topics,num_words), requires_grad=True, device=device)
    #tau, V, B = tau.to(device), V.to(device), B.to(device)

    parameters = chain(encoder.parameters(), decoder.parameters(),[tau,V,B])
    # print('num_words = ', num_words,' num_topics = ', num_topics)
    # print('batch_size = ', batch_size)
    optimizer = optim.Adam(parameters, lr=learning_rate)
    for e in range(num_epoches):
        # print('EPOCH = ', e)
        batches = data.index_batching(batch_size)
        batch_index = 0
        epoch_loss = 0
        for batch in batches:
            if batch_index % 100 == 99:
                print('EPOCH = ', e, ' BATCH = ', batch_index+1)
            doc_batch = [data.docs[d] for d in batch]
            optimizer.zero_grad()
            loss = topicTransformer_loss(encoder, decoder, doc_batch, tau, V, B)
            epoch_loss += loss
            loss /= batch_size
            loss.backward()
            optimizer.step()
            batch_index += 1
        print('EPOCH = ', e, 'loss = ', epoch_loss.item()/data.num_docs)
    top_words = np.argsort(B.cpu().detach().numpy())[:,-20:]
    top_words = top_words[:, ::-1]
    file = open('top_words.txt','w')
    for z in range(num_topics):
        file.write('topic_%d: ' %z)
        for i in range(20):
            j = top_words[z,i]
            file.write(' %s(%f)' %(data.vocabs[j],B[z,j].item()))
        file.write('\n')
    file.close()
if __name__ == "__main__":

    data_file = '../'
    num_topics = 10
    start = time.time()
    main(data_file, num_topics)
    end = time.time()
    print('elapsed time = ', end - start)
