import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V_th = 1 + 0.001 # threshold od spiking neurons
V_th_I = 1
V_reset = 0

def go(y, y0): # update function of the history of spikes / attention
    y[:-1, :] = y[1:, :]
    y[-1, :] = y0
    return y


def RF(delta, g, tau_r): # RF function, equation 14,20 in SI
    g_tmp = np.where(delta<tau_r,g,0)
    g_tmp = np.where(delta==0,1,g_tmp)
    return g_tmp


def sampling(x, DAE1, DAE2, lambda1,lambda2, r1,r2, tau_delta1, tau_delta2, g, tau_r1, tau_r2, tau_1, tau_2, tau_b, tau_t,
              tau_d, decay_rate=0.8, T=3000):
    # parameters are named according to the equations in SI, See table.2 in SI

    W = x.shape[0] # height of the image
    H = x.shape[1] # width of the image
    N = W * H
    d_spk1 = np.zeros((tau_d, N)) # the history of part-level spikes
    d_spk2 = np.zeros((tau_d, N)) # the history of whole-level spikes
    d_top = np.zeros((tau_d, N)) # the history of whole-level spikes, that is needed to compute the top-down feedback

    # initialization of top-down feedback
    gamma1 = np.random.rand(tau_d, N)
    gamma1 = gamma1 / gamma1.sum(0)
    gamma2 = np.random.rand(tau_d, N)
    gamma2 = gamma2/gamma2.sum(0)

    # initialization of delta
    delta1 = np.zeros(N)
    delta2 = np.zeros(N)

    # for ploting: spike recording and attention map
    spk1_record = []
    spk2_record = []
    ga1_record = []
    ga2_record = []
    Ga_record = []

    # for animation
    animation_record = []

    # driving input (original image)
    driv1 = x.reshape(N)
    for t in range(T): # total simulation period
        noise1 = np.random.rand(N) # for realizing bernoulli sampling based on firing rate firing rate
        eps1 = np.random.rand(N) # stochasticity at initial phase(part-level)
        eps2 = np.random.rand(N) # stochasticity at initial phase(whole-level)
        if t % tau_d == 0:          # rapid decay of the stochasticity at initial phase
            r1 = r1 * decay_rate
            r2 = r2 * decay_rate




        # ------------------------------------First block: computing part-level processes------------------------------------------------------------




        feedback = (gamma1[0]) * (d_top[:tau_t, :].sum(0)) + r1 * eps1 # equation 11,22 in SI

        ga_tmp1 = gamma1[0] # to record feedback attention map (inner level)
        Ga_tmp = d_top[:tau_t, :].sum(0) # to record feedback attention map (inter level)

        rf1=RF(delta1, g, tau_r1) # computing the excitability based on the refractory variable delta : equation 14 in SI
        V1 = driv1 * (feedback) * rf1+ noise1 # membrane potential for bernoulli sampling: equation 12 in SI

        spk1 = np.where(V1 > V_th, 1, 0)
        delta1 = delta1- 1 + tau_delta1 * spk1
        delta1 = np.where(delta1 > 0, delta1, 0) # update of spike variable and refractory variable.

        d_spk1 = go(d_spk1, spk1) # update the history of spike pattern (of length tau_d)

        tensor_dspk1 = torch.tensor(d_spk1[-tau_1:, :].sum(0), dtype=torch.float32, device=device) # coincidence detector, integrate the spikes within time window tau_1
                                                                                                   # euqation 16 in SI
        gamma1_now = DAE1(tensor_dspk1).detach().cpu().numpy() # feed to DAE_1 and get the output  # equation 15 in SI
        gamma1 = go(gamma1, gamma1_now)                        # update the history of inner-level top-down attention gamma (of length tau_d)



        # ------------------------------------Second block: computing whole-level processes -----------------------------------------------------------------------------------------------------------------------




        driv2 = lambda1*d_spk1[-tau_b:, :].sum(0) + lambda2* driv1 # the drive of whole level: linear sum of input image and part-level spikes: (first term of) equation 17 in SI
        noise2 = np.random.rand(N) # for realizing bernoulli sampling based on firing rate

        rf2 = RF(delta2, g, tau_r2) # computing the excitability based on the refractory variable delta : equation 20 in SI
        V2 = driv2 * (gamma2[0]+r2*eps2) * rf2 + noise2 # membrane potential for bernoulli sampling: equation 17, 18 in SI
        ga_tmp2 = (gamma2[0]).copy() # record the inner-level top-down attention of whole level

        spk2 = np.where(V2 > V_th, 1, 0)
        delta2 = delta2 - 1 + tau_delta2 * spk2
        delta2 = np.where(delta2 > 0, delta2, 0)  # update of spike variable and refractory variable.

        d_spk2 = go(d_spk2, spk2) # update the history of spike pattern (of length tau_d)
        d_top = go(d_top, d_spk2[-1, :]) # update the relavant spikes to compute (inter level) top-down

        tensor_dspk2 = torch.tensor(d_spk2[-tau_2:, :].sum(0), dtype=torch.float32, device=device) # coincidence detector, integrate the spikes within time window tau_2
                                                                                                   # euqation 16 in SI
        gamma2_now = DAE2(tensor_dspk2).detach().cpu().numpy() # feed to DAE_2 and get the output  # equation 21 in SI
        gamma2 = go(gamma2, gamma2_now) # update the history of inner-level top-down attention gamma2 (of length tau_d)

        #------------------------------------Record some important values for ploting and animation-------------------------------------------------------------------------

        spk1_record.append(spk1)
        spk2_record.append(spk2)
        ga1_record.append(ga_tmp1)
        ga2_record.append(ga_tmp2)
        Ga_record.append(Ga_tmp)

        # record for animation
        animation_record.append(
            [spk1.reshape(W, H), (delta1 / tau_delta1).reshape(W, H), V1.reshape(W, H), feedback.reshape(W, H),
             ga_tmp1.reshape(W, H), d_spk1[-tau_1:, :].sum(0).reshape(W, H), gamma1_now.reshape(W, H),
             (rf1*driv1).reshape(W, H), (rf2*driv1).reshape(W, H), (driv2 * (Ga_tmp)).reshape(W, H), driv2.reshape(W, H),
             ga_tmp2.reshape(W, H), driv1.reshape(W, H), np.zeros((W, H)),
             spk2.reshape(W, H), (delta2 / tau_delta2).reshape(W, H), V2.reshape(W, H), Ga_tmp.reshape(W, H),
             np.zeros((W, H)), d_spk2[-tau_2:, :].sum(0).reshape(W, H), gamma2_now.reshape(W, H)
             ])
    return np.array(spk1_record), np.array(spk2_record), np.array(ga1_record), np.array(ga2_record), animation_record



# np.save('./codeforFigs/'+data+'simdata.npy', [spk1,spk2,ga1,ga2,multi_label2[i], multi_label[i], delay, W, H,i], allow_pickle=True)
