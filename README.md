# Drug_discvovery_pipeline
Model drug id to target (mechanism of action) using 1-D convolution and timestamped sequencing


<h4> Problem statement </h4>
Years ago, therapeutic drugs were derivedfrom  natural  remedies  in  which  the  biologicalmechanisms  were  not  truly  understood.  Evenmore recently, popular drugs such as paraceta-mol,  or  commonly  known  as  Acetaminophen,have been clinically established decades beforetheir  mechanisms  were  understood.  With  theemergence  of  powerful  technologies,  we  areable  to  harness  gene  expression  and  cell  vi-ability  data.  This  data  can  plausibly  be  usedto  measure  patients  cellular  and  genomic  re-sponses,  giving  us  an  indication  of  possibleMechanisms  of  Action  (MoA)  that  may  ex-plain the underlying process in which the drugworks. This targeted method may also provideus  knowledge  of  the  pathology  of  a  givendisease. With this framework, researchers mayidentify relevant protein targets associated witha certain disease, and be able to modulate thisprotein target.
<h4> Data Descrioption</h4>
Our   data   comes   from   a   new   technol-ogy  that  simultaneously  measures  cellular  re-sponses  from  a  pool  of  100  cell  types,  dis-tributed as a part of a Kaggle competition [1].By  pooling  different  cell  types,  we  eliminatethe  need  to  identify  ex-ante–that  is–having  toidentify  which  cells  are  better  suited  for  agiven  drug.  Our  training  data  (see  Figure  1a)is a (23814, 876) frame of data, giving 23,814samples  where  a  dish  of  cells  is  treated  witha  drug.  For  each  of  these  samples  we  havegene  expression  data  (g-0..  ,  g-771)  and  cellviability data (c-0,c-99) telling us the degrada-tion across the 100 different cell lines. We haveapproximately 2,700 drugs, each of which with6 samples that correspond to different dosages and  treatment  duration  times.  Thus,  with  twodosages (D1, D2) and three treatment durations(24,48,72  hrs)  we  have  6  sample  pertubationsfor each drug. 
<br />

<h4> Methodlogy</h4>
The  data  offers  some  problems  however,with   high   variance   of   many   features   evenwithin  the  control  group,  or  being  tested  bythe  same  drug  (see  Figure  2a).  For  example,if  we  take  all  the  samples  associated  with  aparticular drug, we find surprisingly high vari-ance for many genes and cell viability markers,indicating  high-noise  and  randomness  due  to experimentation.


We  attempt  to  reconcile  this  by  collapsing  allsamples of a specific drug into a single (1,876)vector.  If  a  drug  originally  has  6  samples,  weconvert this drugs data from (6,876) to (1,876).We  introduce  convolution  to  collapse  this  di-mension; we learn filters (6x1) for each feature(g-1,g-2... c-99, c-100) in order to more deeplycomprehend  relationships  between  samples  ofthe same drug, collapsing the dimension of the6 dosage and treatment times.

