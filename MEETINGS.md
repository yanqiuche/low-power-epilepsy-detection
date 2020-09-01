### 29-May
Steps to be taken (to get network to work:)

Approaches:
- Check if networks functions correctly by filling network with only 0/1 and then train/check results
- Use only seizure data as training 
- Extract only data from one patient and use that to train (same approach, set everything to 0/1)

### 29-Juny
1. Add to signals and correlation based on that
2. Plot time domain as well
3. Changing of the correlation based on time frames (cross correlation)

### 9-Juli
1. List all things still needed to be done

### 24-Juli
1. Extracted features
2. Subtract to features from each other
3. Check if 50Hz 

### 30-Juli
1. Images + signals under singals
2. Corr_matrix, window, channels

### 4-August
1. Add histograms of the data
2. Show histograms of different time frame B2B
3. Search for networks which use statistical data as input
4. Check if papers about correlation matrix with EC/ normal EEG 


### 13-August
1. Log normal guassian plot
2. Run distrubtion statistics over all data (for loop'ed), different window size (1..10), different patietns
3. Visualize sigma values, 2 histograms of sigma values for different patient, each graph 2 columns seizure/non-seizure, plot all window sizes
4. Check papers (hypothesis; seizure=less correlation)
5. Test statistical relevance based on T-score (https://web.csulb.edu/~msaintg/ppa696/696stsig.htm )
5. 5s and 5s make 1second data and then calculate differences (both correlation matrix and histograms)
    a. If so, create signals based on 1 seconds 