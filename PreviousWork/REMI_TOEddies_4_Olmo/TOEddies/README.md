## Step 1 extraction
Read and save SSH or ADT, U, V, 
Crop to BBOX
Computes du/dx, du/dy, dv/dx, dv/dy
Compute Vorticity  (Vx - Uy)
Compute Speed sqrt(U^2 +V^2)
Compute EKE  (U^2 + V^2)/2
Shear Ss=Vx+Uy
Strain Sn=Ux - Vy
Okubo-Weiss parameter Sn^2+Ss^2 - Vorticity^2
Save to file

## Step 2 Identify eddies 
-- Detect eddies
Identifies local max for Anticyclones and local min for Cyclones
From 3x3 windows it stores the maximum and minimum indexes
Saves the SSH values of the max and min with added noise (SSH_rand)
Save the local max and min for the 'noisy' SSH values, only if it is the only local max (in a 3x3 window)
-- Obtain contours (check Overeal pdf)
Start looking for countours with different levels. Each time it
reduces the amplitude of the contour being searched.  Line 376 Step2
When it finds contours it verifies:
* that they are closed contours
* that they are a closed contour only for one maxima
* that they are at least a square (not too small)

-- After the contours and eddies are detected
Identify centroid
Compute and save EKE, Speed, Vorticity, Center,
