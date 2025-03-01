import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gpflow


sig_e = 1e-4
eps = 1e-4


def K_np(x_data, phi, sigma):
    x_data = np.multiply(x_data, np.reshape(np.exp(phi), [1, -1]))
    dist = np.sum(np.square(x_data), axis=1)
    dist = np.reshape(dist, [-1, 1])
    sq_dists = np.add(np.subtract(dist, 2 * np.matmul(x_data, np.transpose(x_data))), np.transpose(dist))
    my_kernel = np.exp(sigma) * (np.exp(-sq_dists)) + sig_e * np.diag(np.ones(np.shape(x_data)[0]))
    return my_kernel
def K_star_np(x_data, X_M, phi, sigma):
    x_data = np.multiply(x_data, np.reshape(np.exp(phi), [1, -1]))
    X_M = np.multiply(X_M, np.reshape(np.exp(phi), [1, -1]))

    rA = np.reshape(np.sum(np.square(x_data), 1), [-1, 1])
    rB = np.reshape(np.sum(np.square(X_M), 1), [-1, 1])
    pred_sq_dist = np.add(np.subtract(rA, np.multiply(2., np.matmul(x_data, np.transpose(X_M)))), np.transpose(rB))
    pred_kernel = np.exp(sigma) * np.exp(-pred_sq_dist)

    return pred_kernel
def Posterior(x_data, X_M, phi, sigma, y_data, low_bdd, q_mu):
    K_M_0 = np.transpose(K_star_np(x_data, X_M, phi, 0.0))
    K_0 = K_np(x_data, phi, 0.0)
    K_0_inv = np.linalg.inv(K_0)
    GP_mu_M = np.matmul(np.matmul(K_M_0, K_0_inv), y_data)
    ratio_down = (1.0 - np.matmul(np.matmul(K_M_0, K_0_inv), np.transpose(K_M_0)))

    x = np.arange(0, 1, 1.0 / 100.0)
    y = np.arange(0, 1, 1.0 / 100.0)

    X, Y = np.meshgrid(x, y)

    for k in range(X.shape[0]):
        for j in range(Y.shape[0]):
            xx = np.reshape(np.array([X[k, j], Y[k, j]]), [1, 2])
            K_star_0 = np.transpose(K_star_np(x_data, xx, phi, 0.0))
            K_star_M_0 = K_star_np(xx, X_M, phi, 0.0)
            GP_mu_star = np.matmul(np.matmul(K_star_0, K_0_inv), y_data)
            GP_var_star = np.exp(sigma) * (1.0 - np.matmul(np.matmul(K_star_0, K_0_inv), np.transpose(K_star_0)))
            ratio_up = (K_star_M_0 - np.matmul(np.matmul(K_star_0, K_0_inv), np.transpose(K_M_0)))
            GPIO_mean = GP_mu_star + (ratio_up / (ratio_down+eps)) * (q_mu+low_bdd-GP_mu_M)
            GPIO_var = np.maximum(GP_var_star - np.square(ratio_up / (ratio_down+eps)) * (np.exp(sigma) * ratio_down),0.0)
            if k == 0 and j == 0:
                max_coor = xx
                max_value = GPIO_mean + 2.0 * np.sqrt(GPIO_var)
                max_mean=GPIO_mean
                max_var=GPIO_var
            else:
                cur_value = GPIO_mean + 2.0 * np.sqrt(GPIO_var)
                if max_value < cur_value:
                    max_coor = xx
                    max_value = cur_value
                    max_mean=GPIO_mean
                    max_var=GPIO_var
    if np.sum(np.sum(np.abs(x_data-max_coor),axis=1)==0.0) !=0.0:
        max_coor=X_M

    return max_coor,max_mean,max_var
def K(x_data, phi, sigma):
    x_data = tf.multiply(x_data, tf.reshape(tf.exp(phi), [1, -1]))
    dist = tf.reduce_sum(tf.square(x_data), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.add(tf.subtract(dist, 2 * tf.matmul(x_data, tf.transpose(x_data))), tf.transpose(dist))
    my_kernel = tf.exp(sigma) * (tf.exp(-sq_dists)) + sig_e * tf.reshape(tf.diag(tf.ones([1, tf.shape(x_data)[0]])),
                                                                         [tf.shape(x_data)[0], tf.shape(x_data)[0]])
    return my_kernel
def K_star(x_data, X_M, phi, sigma):
    x_data = tf.multiply(x_data, tf.reshape(tf.exp(phi), [1, -1]))
    X_M = tf.multiply(X_M, tf.reshape(tf.exp(phi), [1, -1]))

    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(X_M), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(X_M)))), tf.transpose(rB))
    pred_kernel = tf.exp(sigma) * tf.exp(-pred_sq_dist)
    return pred_kernel
def lbeta(alpha,beta):
    result=tf.lgamma(alpha)+tf.lgamma(beta)-tf.lgamma(alpha+beta)
    return result
def ELBO(x_data,y_data,low_bdd,upp_bdd,X_M,phi,sigma,alpha_q,beta_q,lamda_p):
    eps = 1e-8
    k_vec = K_star(x_data, X_M, phi, 0.0)
    k_vecT = tf.transpose(k_vec)

    R = y_data - (low_bdd) * k_vec
    RT = tf.transpose(R)

    Sig = tf.exp(sigma) * (K(x_data, phi, 0.0) - tf.matmul(k_vec, k_vecT)) + sig_e * tf.reshape(
        tf.diag(tf.ones([1, tf.shape(x_data)[0]])), [tf.shape(x_data)[0], tf.shape(x_data)[0]])
    Sig_inv = tf.matrix_inverse(Sig)

    alpha_pos = tf.exp(alpha_q)
    beta_pos = tf.exp(beta_q)
    lamda_pos = tf.exp(lamda_p)

    KL_div1 = lbeta(1.0,lamda_pos)-lbeta(alpha_pos,beta_pos)+(alpha_pos-1.0)*tf.digamma(alpha_pos)+(beta_pos-lamda_pos)*tf.digamma(beta_pos)
    KL_div2= (1.0+lamda_pos-alpha_pos-beta_pos)*tf.digamma(alpha_pos+beta_pos)

    E1_q = alpha_pos / (alpha_pos + beta_pos)
    E2_q = alpha_pos * beta_pos / (tf.square(alpha_pos + beta_pos) * (alpha_pos + beta_pos + 1)) + tf.square(E1_q)

    KL_div=KL_div1+KL_div2
    Fac1= -0.5 * tf.reduce_sum(tf.log(tf.abs(tf.diag_part(tf.cholesky(Sig))) + eps))- 0.5 * tf.matmul(
        tf.matmul(RT, Sig_inv), R)
    Fac2=E1_q*(upp_bdd-low_bdd)*tf.matmul(tf.matmul(k_vecT,Sig_inv),R)-0.5*E2_q*tf.square(upp_bdd-low_bdd)*tf.matmul(tf.matmul(k_vecT,Sig_inv),k_vec)

    result=Fac1+Fac2-KL_div
    return -result
def acq_fcn_EI(xx,x_data, y_data, low_bdd,upp_bdd, X_M, phi, sigma, q_mu,q_var):
    K_M_0=tf.transpose(K_star(x_data,X_M,phi,0.0))
    K_0 = K(x_data, phi, 0.0)
    K_0_inv = tf.matrix_inverse(K_0)
    GP_mu_M = tf.matmul(tf.matmul(K_M_0, K_0_inv), y_data)
    ratio_down = (1.0 - tf.matmul(tf.matmul(K_M_0, K_0_inv), tf.transpose(K_M_0)))

    K_star_0 = tf.transpose(K_star(x_data, xx, phi, 0.0))
    K_star_M_0 = K_star(xx, X_M, phi, 0.0)
    GP_mu_star = tf.matmul(tf.matmul(K_star_0, K_0_inv), y_data)
    GP_var_star = tf.exp(sigma) * (1.0 - tf.matmul(tf.matmul(K_star_0, K_0_inv), tf.transpose(K_star_0)))
    ratio_up = (K_star_M_0 - tf.matmul(tf.matmul(K_star_0, K_0_inv), tf.transpose(K_M_0)))
    OBCGP_mean = GP_mu_star + (ratio_up / (ratio_down + eps)) * ((upp_bdd - low_bdd) * q_mu + low_bdd - GP_mu_M)
    OBCGP_var = tf.nn.relu(GP_var_star - tf.square(ratio_up / (ratio_down + eps)) * (tf.exp(sigma) * ratio_down))+q_var

    y_max=tf.reduce_max(y_data)
    tau=tf.divide(y_max-OBCGP_mean,tf.sqrt(OBCGP_var)+eps)

    dist = tf.distributions.Normal(loc=0.0,scale=1.0)
    fcn_val= (dist.prob(tau)-tau*dist.cdf(tau))*tf.sqrt(OBCGP_var)

    return fcn_val
def par_update(X, Y,dim):
    with tf.Session() as sess:
        model = gpflow.models.GPR(X, Y, gpflow.kernels.RBF(dim, ARD=False))
        model.clear()
        model.likelihood.variance = sig_e
    
        # model.kern.lengthscales.prior = gpflow.priors.Gamma(1, 1)
        # model.kern.variance.prior = gpflow.priors.Gamma(1., 1.)
    
        model.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(model)

    sigma_np = np.float32(np.log(model.kern.variance.value))
    phi_np = np.reshape(np.float32(0.5 * (-np.log(2.0) - 2.0 * np.log(model.kern.lengthscales.value))), [-1, 1])

    return sigma_np, phi_np