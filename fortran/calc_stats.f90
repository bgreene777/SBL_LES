program calc_stats

implicit none

integer, parameter :: nx = 160, ny = 160, nz = 160
integer, parameter :: nt = 180
integer, parameter :: t0 = 1080000, dnt = 1000 ! starting file and spacing
real(8), parameter :: Lx=800, Ly=800, Lz=400
real(8) :: dx=Lx/dble(nx-1), dy=Ly/dble(ny-1), dz=Lz/dble(nz-1)
real(8) :: z
character(len=128) :: fdir
character(len=128) :: fname, fsave, fsavetke
real(8), dimension(nx, ny, nz, nt) :: u, v, w, theta, dissip, u_rot, v_rot
real(8), dimension(nx, ny, nz, nt) :: Txz, Tyz, q3, zeros=0. !Txx, Tyy, Tzz, q3, zeros=0.
real(8), dimension(nz) :: u_mean, v_mean, w_mean, theta_mean, dissip_mean, &
                          Txz_mean, Tyz_mean, q3_mean, &
                          thetaw_cov_res, thetaw_cov_tot, &
                          uw_cov_res, uw_cov_tot, &
                          vw_cov_res, vw_cov_tot, &
                          u_var_res, u_var_tot, v_var_res, v_var_tot, &
                          w_var_res, w_var_tot, theta_var_res, theta_var_tot,& 
                          uuw, vvw, www
real(8), dimension(nz) :: dudz, dvdz, dwdz, duuwdz, dvvwdz, dwwwdz
real(8), dimension(nz) :: shear_prod, buoy_prod, turb_trans
real(8), parameter :: U_scale = 0.4
real(8), parameter :: T_scale = 300.0
real(8), parameter :: g = 9.81
real(8) :: z_scale = Lz
integer :: i
logical, parameter :: calc_TKE = .true.

fdir = "/home/bgreene/simulations/F_160_interp/output/"
if (.not. calc_TKE) then
    dissip = 0.
    dissip_mean = 0.
end if

! read u, v, w, theta
call load_data(nx,ny,nz,nt,t0,dnt,U_scale,fdir,fname,"u_",u)
call load_data(nx,ny,nz,nt,t0,dnt,U_scale,fdir,fname,"v_",v)
call load_data(nx,ny,nz,nt,t0,dnt,U_scale,fdir,fname,"w_",w)
call load_data(nx,ny,nz,nt,t0,dnt,T_scale,fdir,fname,"theta_",theta)
if (calc_TKE) then
call load_data(nx,ny,nz,nt,t0,dnt,U_scale*U_scale*U_scale/z_scale,&
               fdir,fname,"dissip_",dissip)
end if
! read Txz, Tyz, Txx, Tyy, Tzz, q3
call load_data(nx,ny,nz,nt,t0,dnt,U_scale*U_scale,fdir,fname,"txz_",Txz)
call load_data(nx,ny,nz,nt,t0,dnt,U_scale*U_scale,fdir,fname,"tyz_",Tyz)
!call load_data(nx,ny,nz,nt,t0,dnt,U_scale*U_scale,fdir,fname,"txx_",Txx)
!call load_data(nx,ny,nz,nt,t0,dnt,U_scale*U_scale,fdir,fname,"tyy_",Tyy)
!call load_data(nx,ny,nz,nt,t0,dnt,U_scale*U_scale,fdir,fname,"tzz_",Tzz)
call load_data(nx,ny,nz,nt,t0,dnt,U_scale*T_scale,fdir,fname,"q3_",q3)

!print *, Txx(:,1,1,90)
!print *, Txz(:,1,1,90)

! calculate ubar, vbar, wbar, thetabar
call xytavg(nx, ny, nz, nt, u, u_mean)
call xytavg(nx, ny, nz, nt, v, v_mean)
call xytavg(nx, ny, nz, nt, w, w_mean)
call xytavg(nx, ny, nz, nt, theta, theta_mean)
call xytavg(nx, ny, nz, nt, dissip, dissip_mean)

! rotate coords so <V> = 0; only to be used with calculating variances
call vel_coord_rotate(u, v, u_mean, v_mean, nx, ny, nz, nt, u_rot, v_rot)

! calculate u'w', v'w', and theta'w'
print *, "Begin calculating <u'w'>"
call calc_covariance(nx,ny,nz,nt,u,w,Txz,uw_cov_res,uw_cov_tot)
print *, "Begin calculating <v'w'>"
call calc_covariance(nx,ny,nz,nt,v,w,Tyz,vw_cov_res,vw_cov_tot)
print *, "Begin calculating <theta'w'>"
call calc_covariance(nx,ny,nz,nt,theta,w,q3,thetaw_cov_res,thetaw_cov_tot)

! calculate u'u', v'v', w'w', theta'theta'
print *, "Begin calculating <u'u'>"
call calc_covariance(nx,ny,nz,nt,u_rot,u_rot,zeros,u_var_res,u_var_tot)
print *, "Begin calculating <v'v'>"
call calc_covariance(nx,ny,nz,nt,v_rot,v_rot,zeros,v_var_res,v_var_tot)
print *, "Begin calculating <w'w'>"
call calc_covariance(nx,ny,nz,nt,w,w,zeros,w_var_res,w_var_tot)
print *, "Begin calculating <theta'theta'>"
call calc_covariance(nx,ny,nz,nt,theta,theta,zeros,theta_var_res,theta_var_tot)

!print *, u_var_tot - u_var_res

! write to output csv file
write(fsave,"(a,a)") TRIM(fdir),"average_statistics.csv"
print *, "Saving file: ", fsave

! open file and write csv headers
open(12,file=fsave,action='write',status='replace')
write(12,*) "z, ubar, vbar, wbar, Tbar, dissipbar, uw_cov_res, uw_cov_tot,"//& 
            "vw_cov_res, vw_cov_tot, thetaw_cov_res, thetaw_cov_tot,"//&
            "u_var_res, u_var_tot, v_var_res, v_var_tot,"//&
            "w_var_res, w_var_tot, theta_var_res"
1000 format((F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8,&
             ",",F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8,&
             ",",F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8))

do i=1,nz
    z=(dble(i)-0.5)*dz
    write(12,1000) z,u_mean(i),v_mean(i),w_mean(i),theta_mean(i),dissip_mean(i),&
                   uw_cov_res(i),uw_cov_tot(i),vw_cov_res(i),vw_cov_tot(i),&
                   thetaw_cov_res(i), thetaw_cov_tot(i), u_var_res(i), &
                   u_var_tot(i), v_var_res(i), v_var_tot(i), w_var_res(i), &
                   w_var_tot(i), theta_var_res(i)
end do
close(12)

if (calc_TKE) then
    print *, "Begin TKE budget terms"
    ! Now begin calculating TKE budget terms
    ! Only do z-terms (j=3)
    ! (1) Shear Production: -<vi'vj'> d<vi>/dxj -> -<u'w'>d<u>/dz -<v'w'>d<v>/dz -<w'w'>d<w>/dz
    call calc_ddz(nz, dz, u_mean, dudz)
    call calc_ddz(nz, dz, v_mean, dvdz)
    call calc_ddz(nz, dz, w_mean, dwdz)

    shear_prod = -uw_cov_tot*dudz -vw_cov_tot*dvdz -w_var_tot*dwdz

    ! (2) Buoyancy Production: beta <w'theta'>, beta=g/theta0
    buoy_prod = (g/theta_mean(1)) * thetaw_cov_tot

    ! (3) Turbulent Transport: -d<e'vj'>/dxj -> -d<e'w'>/dz = -(1/2)[d<u'u'w'>/dz + d<v'v'w'>/dz + d<w'w'w'>/dz]
    call calc_3order(nx, ny, nz, nt, u, u, w, uuw)
    call calc_3order(nx, ny, nz, nt, v, v, w, vvw)
    call calc_3order(nx, ny, nz, nt, w, w, w, www)

    call calc_ddz(nz, dz, uuw, duuwdz)
    call calc_ddz(nz, dz, vvw, dvvwdz)
    call calc_ddz(nz, dz, www, dwwwdz)

    turb_trans = -0.5 * (duuwdz + dvvwdz + dwwwdz)

    ! (4) Dissipation Rate: Dissip (term calculated above is sum of individual components)

    ! write to output csv file
    write(fsavetke,"(a,a)") TRIM(fdir),"tke_budget.csv"
    print *, "Saving file: ", fsavetke
    ! open file and write csv headers
    open(13,file=fsavetke,action='write',status='replace')
    write(13,*) "z, shear, buoyancy, turbtrans, dissipation3d"

    1001 format((F12.8,",",F12.8,",",F12.8,",",F12.8,",",F12.8))

    do i=1,nz
        z=(dble(i)-0.5)*dz
        write(13,1001) z, shear_prod(i), buoy_prod(i), turb_trans(i), dissip_mean(i)
    end do
    close(13)
end if

print *, "Done!"

end program calc_stats

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
subroutine calc_covariance(nx, ny, nz, nt, var1, var2, sgs, var1var2_cov_res, &
                           var1var2_cov_tot)

implicit none

! declare input params
integer :: nx, ny, nz, nt
real(8), dimension(nx, ny, nz, nt) :: var1, var2, sgs
! declare output
real(8), dimension(nz) :: var1_xytavg, var2_xytavg, var1var2_cov_res, &
                          var1var2_cov_tot
! declare variables to be used
integer :: i, j, k, l
real(8), dimension(nx, ny, nz, nt) :: var1_fluc, var2_fluc, var1var2_fluc
real(8), dimension(nz) :: sgs_xytavg

! first calculate xytavg for var1 and var2
call xytavg(nx,ny,nz,nt,var1,var1_xytavg)
call xytavg(nx,ny,nz,nt,var2,var2_xytavg)
! calculate xytavg for sgs
call xytavg(nx,ny,nz,nt,sgs,sgs_xytavg)

! calculate fluctuating components
print *, "Beginning calculating fluctuating components"
do k=1, nz
    var1_fluc(:,:,k,:) = var1(:,:,k,:) - var1_xytavg(k)
    var2_fluc(:,:,k,:) = var2(:,:,k,:) - var2_xytavg(k)
end do
print *, "Finished calculating fluctuating components"

! calculate instantaneous co-fluctuations
var1var2_fluc = var1_fluc * var2_fluc

! average the instantaneous co-fluctuations
call xytavg(nx,ny,nz,nt,var1var2_fluc,var1var2_cov_res)

! calculate total (resolved + SGS) covariances
var1var2_cov_tot = var1var2_cov_res + sgs_xytavg

end subroutine calc_covariance

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine calc_3order(nx, ny, nz, nt, var1, var2, var3, var1var2var3_cov)

implicit none

! declare input params
integer :: nx, ny, nz, nt
real(8), dimension(nx, ny, nz, nt) :: var1, var2, var3
! declare output
real(8), dimension(nz) :: var1var2var3_cov
! declare variables to be used
integer :: i, j, k, l
real(8), dimension(nx, ny, nz, nt) :: var1_fluc, var2_fluc, var3_fluc, &
                                      var1var2var3_fluc
real(8), dimension(nz) :: var1_xytavg, var2_xytavg, var3_xytavg

! first calculate xytavg for var1, var2, var3
call xytavg(nx,ny,nz,nt,var1,var1_xytavg)
call xytavg(nx,ny,nz,nt,var2,var2_xytavg)
call xytavg(nx,ny,nz,nt,var3,var3_xytavg)

! calculate fluctuating components
print *, "Beginning calculating fluctuating components"
do k=1, nz
    var1_fluc(:,:,k,:) = var1(:,:,k,:) - var1_xytavg(k)
    var2_fluc(:,:,k,:) = var2(:,:,k,:) - var2_xytavg(k)
    var3_fluc(:,:,k,:) = var3(:,:,k,:) - var3_xytavg(k)
end do
print *, "Finished calculating fluctuating components"

! calculate instantaneous co-fluctuations
var1var2var3_fluc = var1_fluc * var2_fluc * var3_fluc

! average the instantaneous co-fluctuations
call xytavg(nx,ny,nz,nt,var1var2var3_fluc,var1var2var3_cov)

end subroutine calc_3order

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine calc_ddz(nz, dz, var, dvardz)

implicit none

! declare input params
integer :: nz
real(8) :: dz
real(8), dimension(nz) :: var
! declare output params
real(8), dimension(nz) :: dvardz
! declare variables to be used
integer :: k
real(8) :: dvar

dvardz(1) = 0.
dvardz(nz) = 0.

do k=2, nz-1
    dvar = var(k+1) - var(k-1)
    dvardz(k) = dvar/(2.0 * dz)
end do

end subroutine calc_ddz

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