program corr2d

implicit none

! allocate parameters and variables
integer, parameter :: nx = 192, ny = 192, nz = 192
integer, parameter :: nt = 180
integer, parameter :: t0 = 1080000, dnt = 1000 ! starting file and spacing
real(8), parameter :: Lx=800, Ly=800, Lz=400
real(8) :: dx=Lx/dble(nx), dy=Ly/dble(ny), dz=Lz/dble(nz)
real(8) :: z
character(len=128) :: fdir
character(len=128) :: fname, fsave
real(8), dimension(nx, ny, nz, nt) :: u, v, w, theta, u_rot, v_rot
real(8), dimension(nx, ny, nz) :: var_temp
real(8), dimension(nz) :: u_mean, v_mean, w_mean, theta_mean,&
                          u_mean_rot, v_mean_rot
real(8), parameter :: U_scale = 0.4, T_scale = 300.0

! get nmax and fdir from command line arguments
! 2 arguments: fdir and nmax
integer, parameter :: k_start=1
integer :: nmax
real(8), dimension(:,:), allocatable :: corr2d_all, corr
integer :: jt
integer :: num_args, ix
character(len=192), dimension(:), allocatable :: args

! get command line inputs
num_args = command_argument_count()
allocate(args(num_args))  ! I've omitted checking the return status of the allocation 

do ix = 1, num_args
    call get_command_argument(ix,args(ix))
end do
read(args(1),"(a)") fdir
read(args(2),*) nmax

! allocate corr2d_all and corr
allocate(corr2d_all(-nmax:nmax,0:nmax))
allocate(corr(-nmax:nmax,0:nmax))

print *, "Shape of corr2d_all: ", shape(corr2d_all)
print *, "nmax = ", nmax

! fdir = "/home/bgreene/simulations/A.10_192_interp/output/"
print *, "fdir: ", fdir

! begin loading data
! read u, v, w, theta
call load_data(nx,ny,nz,nt,t0,dnt,U_scale,fdir,fname,"u_",u)
call load_data(nx,ny,nz,nt,t0,dnt,U_scale,fdir,fname,"v_",v)
! call load_data(nx,ny,nz,nt,t0,dnt,U_scale,fdir,fname,"w_",w)
! call load_data(nx,ny,nz,nt,t0,dnt,T_scale,fdir,fname,"theta_",theta)

! calculate ubar, vbar, wbar, thetabar
print *, "Begin calculating mean quantities..."
call xytavg(nx, ny, nz, nt, u, u_mean)
call xytavg(nx, ny, nz, nt, v, v_mean)
! call xytavg(nx, ny, nz, nt, w, w_mean)
! call xytavg(nx, ny, nz, nt, theta, theta_mean)

! rotate coords so <V> = 0; only to be used with calculating variances
print *, "Begin rotating cordinates..."
call vel_coord_rotate(u, v, u_mean, v_mean, nx, ny, nz, nt, u_rot, v_rot)

! calculate mean rotated velocities
call xytavg(nx, ny, nz, nt, u_rot, u_mean_rot)
call xytavg(nx, ny, nz, nt, v_rot, v_mean_rot)

! loop over timesteps and calculate correlations
print *, "Begin calculating correlations..."
do jt=1, nt
    var_temp = u_rot(:,:,:,jt)
    call calc_2d_corr_xz(k_start, nx, ny, nz, dx, dz, var_temp, nmax, corr)
    corr2d_all(:,:) = corr2d_all(:,:) + corr(:,:)
end do

! average in time
corr2d_all = corr2d_all / dble(nt)

! save file
print *, "Shape of corr2d_all: ", shape(corr2d_all)
write(fsave,"(a,a,a)") TRIM(fdir), "netcdf/", "R_uu.out"
print *, "Saving file: ", fsave

!R_uu
open(12,file=TRIM(fsave),access="direct",status="unknown",recl=8*size(corr2d_all))
write(12,rec=1) corr2d_all
close(12)

end program corr2d

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine load_data(nx, ny, nz, nt, t0, dnt, scale, fdir, fname, cvar, x)

implicit none

! delcare input params
integer :: nx, ny, nz, nt, t0, dnt
real(8) :: scale
character(len=128) :: fdir
character(len=128) :: fname
character(*) :: cvar
! declare output
real(8), dimension(nx, ny, nz, nt) :: x
! declare variables to be used
integer :: i
integer, dimension(nt) :: timesteps

do i=1, nt
    timesteps(i) = t0 + (i*dnt)
end do

do i=1, nt
    write(fname,"(a,a,i7.7,a)") TRIM(fdir),cvar,timesteps(i),".out"
    print *, "Reading file: ", fname
    open(10,file=TRIM(fname),form="unformatted",access="stream",status="old")
    read(10) x(:,:,:,i)
    close(10)
end do

x = x * scale

end subroutine load_data

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine xytavg(nx, ny, nz, nt, dat, dat_xytavg)

implicit none

! declare input params
integer :: nx, ny, nz, nt
real(8), dimension(nx, ny, nz, nt) :: dat
! declare output
real(8), dimension(nz) :: dat_xytavg
! declare variables to be used
integer :: i

! average in x-y-t slices
do i=1, nz
    dat_xytavg(i) = SUM(dat(:,:,i,:)) / dble(nx*ny*nt)
end do

!print *, dat_xytavg

end subroutine xytavg

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine vel_coord_rotate(u, v, u_mean, v_mean, nx, ny, nz, nt, u_rot, v_rot)

    implicit none
    
    ! declare input params
    integer :: nx, ny, nz, nt
    real(8), dimension(nx, ny, nz, nt) :: u, v
    real(8), dimension(nz) :: u_mean, v_mean
    ! declare output params
    real(8), dimension(nx, ny, nz, nt) :: u_rot, v_rot
    ! declare variables to be used
    integer :: ix, jy, kz, it
    real(8) :: angle, ca, sa
    
    ! begin big loop
    ! start with looping over kz to get angle from mean u and v
    do kz=1,nz
        angle = datan2(v_mean(kz), u_mean(kz))
        ! calc sin and cos of angle once here for efficiency
        ca = cos(angle)
        sa = sin(angle)
        ! now loop through x, y, t
        do ix=1,nx
        do jy=1,ny
        do it=1,nt
            u_rot(ix,jy,kz,it) = u(ix,jy,kz,it)*ca + v(ix,jy,kz,it)*sa
            v_rot(ix,jy,kz,it) =-u(ix,jy,kz,it)*sa + v(ix,jy,kz,it)*ca
        end do
        end do
        end do
    end do
    
    print *, "Finished rotating coordinates!"
    
end subroutine vel_coord_rotate

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine calc_2d_corr_xz(k_start,nx,ny,nz,dx,dz,var,nmax,corr2d)

  implicit none
  
  ! variables
  ! input
  !Input Variables
  integer :: nx,ny,nz,k_start,nmax
  real(kind=8) :: dx,dz
  real(kind=8),dimension(nx,ny,nz) :: var
  
  !Output Variables
  real(kind=8),dimension(-nmax:nmax,0:nmax) :: corr2d
  
  !Local variables
  integer :: i,j,k,jx,jy,jz
  integer :: ixlag,izlag
  real(kind=8) :: a_avg,b_avg
  real(kind=8) :: sigma_a,sigma_b
  integer :: counter
  real(kind=8),dimension(-nmax:nmax) :: xlags
  real(kind=8),dimension(0:nmax) :: zlags

  corr2d(:,:) = 0.0

  a_avg=SUM(var(:,:,k_start))/(nx*ny)
  sigma_a=SQRT(SUM((var(:,:,k_start)-a_avg)**2)/(nx*ny))
!   write(*,*)'a_avg=',a_avg

  do k=0,nmax
    b_avg=SUM(var(:,:,k_start+k))/(nx*ny)
    sigma_b=SQRT(SUM((var(:,:,k_start+k)-b_avg)**2)/(nx*ny))
    ! write(*,*)'k,b_avg',k,b_avg

  do i=-nmax,nmax

    counter=0

    do jx=1,nx
    do jy=1,ny

      ixlag=jx+i
      izlag=k_start+k

      if (ixlag > nx) then
        ixlag=ixlag - (nx-1)
      elseif (ixlag<1) then
        ixlag=ixlag+(nx-1)
      endif

      corr2d(i,k) = corr2d(i,k)+(var(jx,jy,k_start)-a_avg)*(var(ixlag,jy,izlag)-b_avg)
      counter=counter+1

    enddo
    enddo

    !Divide by counter
    corr2d(i,k)=corr2d(i,k)/real(counter,8)

  enddo
  enddo

  !Normalize by standard deviations at two heights of interest
  a_avg=SUM(var(:,:,k_start))/(nx*ny)
  sigma_a=SQRT(SUM((var(:,:,k_start)-a_avg)**2)/(nx*ny))
  do k=0,nmax
    b_avg=SUM(var(:,:,k_start+k))/(nx*ny)
    sigma_b=SQRT(SUM((var(:,:,k_start+k)-b_avg)**2)/(nx*ny))
    corr2d(:,k)=corr2d(:,k)/(sigma_a*sigma_b)
  enddo

end subroutine calc_2d_corr_xz