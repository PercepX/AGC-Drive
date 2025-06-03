# Iterative Closest Point (ICP) Algorithm in Python
An implementation of Iterative Closest Point Algorithm in Python based on *Besl, P.J. & McKay, N.D. 1992, 'A Method for Registration of 3-D Shapes', IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 14, no. 2,  IEEE Computer Society*. 


## Usage
The code can be run as follows:
```
git clone https://github.com/iitaakash/icp_python.git
pip3 install -r requirements.txt
python3 main.py 
```

For using ICP on your dataset see the icp.py file. The usage is as follows:

`(R, t) = IterativeClosestPoint(source_pts, target_pts, tau)` where R and t are the estimated rotation and translation using ICP between the source points and the target points. tau is the threshold to terminate the algorithm. It terminates when the change in RMSE is less than tau between two successive iterations.
