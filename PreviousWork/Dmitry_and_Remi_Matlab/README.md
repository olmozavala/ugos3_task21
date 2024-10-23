### Email 

The function for LC/LCE identificaiton is here:

```shell
ddmitry@enterprise: /distance_metric> pwd
/home/ddmitry/codes/anls_mtlb_utils/hycom_TSIS/distance_metric
ddmitry@enterprise: /distance_metric> ll identify_LC.m 
```

It automatically detects both LC and all anticyclonic eddies based on 
the selected ssh contour = Bisol (which is 0.17 m, typically). 
Note that before running the code, SSH in the Gulf has to be demeaned ! 
You can see how I do it here, for example:

```shell
```
ddmitry@enterprise: /hycom_TSIS> pwd
/home/ddmitry/codes/anls_mtlb_utils/hycom_TSIS

extr_lc_hycom_nemo-V0.m

```Matlab
  fprintf('Reading %s\n',fina);
  fld = 'srfhgt';
  [F,nn,mm,ll] = read_hycom(fina,finb,fld);
  F(F>huge)=nan;
  ssh=squeeze(F)./(1e-3*rg);  % ssh m
%
% Subtract anomaly:
  dmm=ssh;
  dmm(INH==0)=nan;
%  dmm(HH>-200)=nan;
  sshM=nanmean(nanmean(dmm));
  ssh=ssh-sshM;

%
% Derive LC contour:
% 
  dmm=ssh;
  dmm(INH==0)=nan;
  LCH2 = identify_LC(LON,LAT,dmm,Bisol);
```

I also have an algorithm for cyclone detection (it works slightly different). 
Remi's code is here:

```shell
ddmitry@enterprise: /LC_front_Remi> pwd
/home/ddmitry/codes/anls_mtlb_utils/hycom_TSIS/LC_front_Remi
```

Let me know if you have questions.

Best,
Dmitry
