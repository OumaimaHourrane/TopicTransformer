import torch

def doc_batch_to_vec_tensor(doc_batch, num_words, batch_size, stopwords = None):
    """
    convert a batch of docs' word vector into tensor
    :param doc_batch:
    :param batch_size:
    :param word_indexes:
    :param num_words:
    :return:
    """
    assert len(doc_batch) == batch_size, "#docs = %d, batch_size = %d" %(len(doc_batch), batch_size)
    doc_tensor = torch.zeros(batch_size,num_words).long()
    for d in range(batch_size):
        for i in range(len(doc_batch[d])):
            w = doc_batch[d][i]
            if stopwords == None:
                doc_tensor[d,w] += 1
            else:
                if w not in stopwords:
                    doc_tensor[d, w] += 1
    return doc_tensor

def doc_batch_to_seg_tensor(doc_batch, batch_size):
    """
    convert a batch of docs' word sequence into tensor
    :param doc_batch:
    :param batch_size:
    :param word_indexes:
    :param num_words:
    :return:
    """
    assert len(doc_batch) == batch_size, "#docs = %d, batch_size = %d" % (len(doc_batch), batch_size)
    doc_lengths = torch.LongTensor([len(doc_batch[d]) for d in range(batch_size)])
    max_length = max(doc_lengths)
    doc_tensor = torch.zeros(batch_size,max_length).long()
    for d in range(batch_size):
        doc_tensor[d,:doc_lengths[d]] = torch.LongTensor(doc_batch[d])
    return doc_tensor, doc_lengths

def reparameterize(mu, log_sigma):
    std = torch.exp(0.5 * log_sigma)
    # eps = torch.randn_like(std)
    eps = torch.zeros_like(std)
    return eps.mul(std).add_(mu)


def kld(mus, log_sigmas):
    kl_distance = -0.5 * torch.sum(1 + log_sigmas - mus.pow(2) - log_sigmas.exp())
    return kl_distance

def recovery_loss(doc_batch, thetas):
    return 0

def sparsity_regularization(thetas):
    return -torch.sum(thetas*thetas.log())



