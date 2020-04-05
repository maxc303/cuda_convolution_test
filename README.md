# cuda_conv
<h2>Cudnn</h2> 
conv_cudnn/: solution of cudnn <br />
on GTX960m <br />
4000x3000.jpg: 0.0550 <br />
2000x1000.jpg: 0.00891 <br />
780x585.jpg: 0.00220 <br /><br />

<h2>CPU Solution conv_cpu.cu</h2> 
CPU: i7-6700HQ<br />
4000x3000.jpg: 7.45588 <br />
2000x1000.jpg: 1.23983 <br /> 
780x585.jpg: 0.27348 <br />

<h2>CUDA naive Solution conv_cuda_naive.cu</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.1769 <br />
2000x1000.jpg: 0.0346<br /> 
780x585.jpg: 0.00728 <br />

<h2>CUDA constant memory Solution conv_cuda_cmem.cu</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.1725 <br />
2000x1000.jpg: 0.0338<br /> 
780x585.jpg: 0.00706 <br />

<h2>CUDA register output, no padding conv_reg_nopad.cu</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0799 <br />
2000x1000.jpg: 0.0129<br /> 
780x585.jpg: 0.00304 <br />

<h2>CUDA register output, CMEM no padding conv_reg_cmem_nopad.cu</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0763 <br />
2000x1000.jpg: 0.0125<br /> 
780x585.jpg: 0.00296 <br />

<h2>CUDA register output, padded </h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0813 <br />
2000x1000.jpg: 0.0158<br /> 
780x585.jpg: 0.00331 <br />

<h2>CUDA SMEM in , register output, cmem kernel 32x32 conv_cuda_smem_nopad.cu</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0651 <br />
2000x1000.jpg: 0.0110<br /> 
780x585.jpg: 0.00258 <br />

<h2>CUDA SMEM in , register output, cmem kernel 32x16 conv_cuda_smem_nopad.cu</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0587 <br />
2000x1000.jpg: 0.0103<br /> 
780x585.jpg: 0.00241 <br />

<h2>CUDA SMEM in , register output, cmem kernel 32x16 ,reorder SMEM conv_cuda_smem_reorder.cu</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0540 <br />
2000x1000.jpg: 0.00899<br /> 
780x585.jpg: 0.00216 <br />

<h2>CUDA SMEM in , register output, cmem kernel 32x16 ,reorder SMEM, conv_cuda_smem_reorder.cu </h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0510 <br />
2000x1000.jpg: 0.00899<br /> 
780x585.jpg: 0.00216 <br />


<h2>CUDA SMEM in , register output, cmem kernel 32x32 ,reorder SMEM </h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0577 <br />
2000x1000.jpg: 0.00917<br /> 
780x585.jpg: 0.00214 <br />


<h2>CUDA SMEM in , register output, cmem kernel 32x16 ,reorder SMEM ,combine some copy function conv_cuda_smem_combine</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0488<br />
2000x1000.jpg: 0.00841<br /> 
780x585.jpg: 0.00195 <br />

<h2>CUDA SMEM in , register output, cmem kernel 64x16 ,reorder SMEM ,combine some copy function</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0492<br />
2000x1000.jpg: 0.00855<br /> 
780x585.jpg: 0.00204 <br />

<h2>CUDA SMEM in , register output, cmem kernel 32x32 ,reorder SMEM ,combine top and left, bottom and right</h2> 
on GTX960m kernel time<br />
4000x3000.jpg: 0.0477<br />
2000x1000.jpg: 0.00813<br /> 
780x585.jpg: 0.00185<br />





