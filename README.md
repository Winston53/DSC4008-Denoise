## Work Flow
- Build Dataset -> Using BNP method for building the burst and adding noise -> training the NAFNet
1. download_sample.py
2. train.py

## Dataset
- 20 000 distinct photos for training
- Seed = 4008
- Size condidtion: H or W must >= 640 pixels
- Perform the test like Burstormer

## Data Preprorocessing
1. Sourse images from `Open Image V6`
2. Burst geometry with using a Poisson-based probability
3. Exposure and noise with a log-uniform white level and Gaussian read and signal-dependent shot noise
4. Training pair
NOTE: Burst geometry, exposure and noise is referencing `BNP` [Link](https://github.com/likesum/bpn/blob/main/utils/tf_utils.py)


## NAFNet
- Input: Noisy frame 0
- 300 epochs
- Learning rate = 1e-4
- L1 and Adam