# cuda_conv
<h2>Cudnn</h2> 
conv_cudnn/: solution of cudnn <br />
on GTX960m <br />
4000x3000.jpg: 0.0550 <br />
2000x1000.jpg: 0.00891 <br />
780x585.jpg: 0.00220 <br /><br />

<h2>CPU Solution</h2> 
CPU: i7-6700HQ<br />
4000x3000.jpg: 7.45588 <br />
2000x1000.jpg: 1.23983 <br /> 
780x585.jpg: 0.27348 <br />

<h2>CUDA naive Solution</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.1769 <br />
2000x1000.jpg: 0.0346<br /> 
780x585.jpg: 0.00728 <br />

<h2>CUDA constant memory Solution</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.1725 <br />
2000x1000.jpg: 0.0338<br /> 
780x585.jpg: 0.00706 <br />


