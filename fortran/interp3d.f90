PROGRAM interp3d

IMPLICIT NONE

! define same kind as in abl_les
integer, parameter :: rprec = kind (1.d0)

INTEGER,PARAMETER :: nx = 96
INTEGER,PARAMETER :: ny = 96
INTEGER,PARAMETER :: nz = 97
INTEGER,PARAMETER :: lh=nx/2+1,ld=2*lh
INTEGER,PARAMETER :: nx_new = 256
INTEGER,PARAMETER :: ny_new = 256
INTEGER,PARAMETER :: nz_new = 256
INTEGER,PARAMETER :: lh_new=nx_new/2+1,ld_new=2*lh_new
REAL(KIND=rprec) :: lx, ly, lz, lz0

CHARACTER(len=128) :: fname
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: u
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: v
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: w
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: theta
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: RHSx
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: RHSy
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: RHSz
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: RHS_T
REAL(KIND=rprec),DIMENSION(ld,ny,1)  :: sgs_t3
REAL(KIND=rprec),DIMENSION(ld,ny)    :: sgs_t3_2d
REAL(KIND=rprec),DIMENSION(nx,ny)    :: psi_m
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: Cs_opt2
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: F_LM
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: F_MM
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: F_QN
REAL(KIND=rprec),DIMENSION(ld,ny,nz) :: F_NN
REAL(KIND=rprec),DIMENSION(nx,ny)    :: T_s_init

REAL(KIND=rprec),DIMENSION(nx) :: xx
REAL(KIND=rprec),DIMENSION(ny) :: yy
REAL(KIND=rprec),DIMENSION(nz) :: zz
REAL(KIND=rprec),DIMENSION(nx_new,ny_new,nz_new) :: u_new, v_new, w_new, theta_new, &
                                                RHSx_new, RHSy_new, RHSz_new, &
                                                RHS_T_new, Cs_opt2_new, F_LM_new, &
                                                F_MM_new, F_QN_new, F_NN_new
! same variables but with x dimension of ld_new for saving
REAL(KIND=rprec),DIMENSION(ld_new,ny_new,nz_new+1) :: u_new_ld, v_new_ld, w_new_ld, theta_new_ld, &
                                                RHSx_new_ld, RHSy_new_ld, RHSz_new_ld, &
                                                RHS_T_new_ld, Cs_opt2_new_ld, F_LM_new_ld, &
                                                F_MM_new_ld, F_QN_new_ld, F_NN_new_ld 
REAL(KIND=rprec),DIMENSION(ld_new,ny_new,1) :: sgs_t3_new
REAL(KIND=rprec),DIMENSION(nx_new,ny_new) :: psi_m_new, sgs_t3_2d_new, T_s_init_new
REAL(KIND=rprec),DIMENSION(nx_new) :: xx_new
REAL(KIND=rprec),DIMENSION(ny_new) :: yy_new
REAL(KIND=rprec),DIMENSION(nz_new) :: zz_new
REAL(KIND=rprec),DIMENSION(nx_new,ny_new,1) :: theta_new_2d

character(len=128) :: fout, fout2, fout3, fout4, fout5, fout6, fout7      !Name of output files

fname='/home/bgreene/simulations/F_256_interp/vel_sc_spinup.out'
print*,"Reading file: ", fname
open(10,file=TRIM(fname),form='unformatted')
READ (10) u(:,:,1:nz),v(:,:,1:nz),w(:,:,1:nz), theta(:,:,1:nz),   &
          RHSx(:,:,1:nz), RHSy(:,:,1:nz), RHSz(:,:,1:nz), RHS_T(:,:,1:nz), &
          sgs_t3(:,:,1), psi_m, Cs_opt2(:,:,1:nz), F_LM(:,:,1:nz), &
          F_MM(:,:,1:nz), F_QN(:,:,1:nz), F_NN(:,:,1:nz), T_s_init
                
print *,"Shape of sgs_t3: ", SHAPE(sgs_t3)
print *,"Shape of psi_m: ", SHAPE(psi_m)
print *,"Shape of Cs_opt2: ", SHAPE(Cs_opt2)
print *,"Shape of F_LM: ", SHAPE(F_LM)
print *,"Shape of F_MM: ", SHAPE(F_MM)
print *,"Shape of F_QN: ", SHAPE(F_QN)
print *,"Shape of F_NN: ", SHAPE(F_NN)

! sgs_t3 is (nx, ny, 1) so convert to 2d for interpolating then can add 3d after
sgs_t3_2d(:,:) = sgs_t3(:,:,1)

lx = 800.
ly = 800.
lz = 400.

! u
print *,"Begin interpolating u"
call interpolate(nx, ny, nz-1, lx, ly, lz, u(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, u_new)
u_new_ld(:,:,:) = 0.
u_new_ld(1:nx_new,:,1:nz_new) = u_new(:,:,:)
print *,"Finished interpolating u"
! v
print *,"Begin interpolating v"
call interpolate(nx, ny, nz-1, lx, ly, lz, v(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, v_new)
v_new_ld(:,:,:) = 0.
v_new_ld(1:nx_new,:,1:nz_new) = v_new(:,:,:)
print *,"Finished interpolating v"
! w
print *,"Begin interpolating w"
call interpolate(nx, ny, nz-1, lx, ly, lz, w(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, w_new)
w_new_ld(:,:,:) = 0.
w_new_ld(1:nx_new,:,1:nz_new) = w_new(:,:,:)
print *,"Finished interpolating w"
! theta
print *,"Begin interpolating theta"
call interpolate(nx, ny, nz-1, lx, ly, lz, theta(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, theta_new)
theta_new_ld(:,:,:) = 0.
theta_new_ld(1:nx_new,:,1:nz_new) = theta_new(:,:,:)
print *,"Finished interpolating theta"
! RHSx
print *,"Begin interpolating RHSx"
call interpolate(nx, ny, nz-1, lx, ly, lz, RHSx(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, RHSx_new)
RHSx_new_ld(:,:,:) = 0.
RHSx_new_ld(1:nx_new,:,1:nz_new) = RHSx_new(:,:,:)
print *,"Finished interpolating RHSx"
! RHSy
print *,"Begin interpolating RHSy"
call interpolate(nx, ny, nz-1, lx, ly, lz, RHSy(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, RHSy_new)
RHSy_new_ld(:,:,:) = 0.
RHSy_new_ld(1:nx_new,:,1:nz_new) = RHSy_new(:,:,:)
print *,"Finished interpolating RHSy"
! RHSz
print *,"Begin interpolating RHSz"
call interpolate(nx, ny, nz-1, lx, ly, lz, RHSz(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, RHSz_new)
RHSz_new_ld(:,:,:) = 0.
RHSz_new_ld(1:nx_new,:,1:nz_new) = RHSz_new(:,:,:)
print *,"Finished interpolating RHSz"
! RHS_T
print *,"Begin interpolating RHS_T"
call interpolate(nx, ny, nz-1, lx, ly, lz, RHS_T(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, RHS_T_new)
RHS_T_new_ld(:,:,:) = 0.
RHS_T_new_ld(1:nx_new,:,1:nz_new) = RHS_T_new(:,:,:)
print *,"Finished interpolating RHS_T"
! sgs_t3
print *,"Begin interpolating sgs_t3"
call interpolate2d(nx, ny, lx, ly, sgs_t3_2d(1:nx,:), nx_new, ny_new, sgs_t3_2d_new)
sgs_t3_new(1:nx_new,:,1) = sgs_t3_2d_new(:,:) ! add back the 3rd dimension
print *,"Finished interpolating sgs_t3"
! psi_m
print *,"Begin interpolating psi_m"
call interpolate2d(nx, ny, lx, ly, psi_m(1:nx,:), nx_new, ny_new, psi_m_new)
print *,"Finished interpolating psi_m"
! Cs_opt2
print *,"Begin interpolating Cs_opt2"
call interpolate(nx, ny, nz-1, lx, ly, lz, Cs_opt2(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, Cs_opt2_new)
Cs_opt2_new_ld(:,:,:) = 0.
Cs_opt2_new_ld(1:nx_new,:,1:nz_new) = Cs_opt2_new(:,:,:)
print *,"Finished interpolating Cs_opt2"
! F_LM
print *,"Begin interpolating F_LM"
call interpolate(nx, ny, nz-1, lx, ly, lz, F_LM(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, F_LM_new)
F_LM_new_ld(:,:,:) = 0.
F_LM_new_ld(1:nx_new,:,1:nz_new) = F_LM_new(:,:,:)
print *,"Finished interpolating F_LM"
! F_MM
print *,"Begin interpolating F_MM"
call interpolate(nx, ny, nz-1, lx, ly, lz, F_MM(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, F_MM_new)
F_MM_new_ld(:,:,:) = 0.
F_MM_new_ld(1:nx_new,:,1:nz_new) = F_MM_new(:,:,:)
print *,"Finished interpolating F_MM"
! F_QN
print *,"Begin interpolating F_QN"
call interpolate(nx, ny, nz-1, lx, ly, lz, F_QN(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, F_QN_new)
F_QN_new_ld(:,:,:) = 0.
F_QN_new_ld(1:nx_new,:,1:nz_new) = F_QN_new(:,:,:)
print *,"Finished interpolating F_QN"
! F_NN
print *,"Begin interpolating F_NN"
call interpolate(nx, ny, nz-1, lx, ly, lz, F_NN(1:nx,:,1:nz-1), nx_new, ny_new, nz_new, F_NN_new)
F_NN_new_ld(:,:,:) = 0.
F_NN_new_ld(1:nx_new,:,1:nz_new) = F_NN_new(:,:,:)
print *,"Finished interpolating F_NN"
! T_s_init
print *,"Begin interpolating T_s_init"
call interpolate2d(nx, ny, lx, ly, T_s_init(1:nx,:), nx_new, ny_new, T_s_init_new)
print *,"Finished interpolating T_s_init"

print *,"All parameters finished interpolating!"
!print*,"Shape of u: ", SHAPE(u)
!print*,"Shape of u_new: ", SHAPE(u_new)

! save interpolated u field
!fout='/home/bgreene/fortran/u_interp.out'
!print*,"Saving file: ", fout
!open(1001,file=TRIM(fout),access='direct',status='unknown',recl=8*nx_new*ny_new*nz_new)
!    write(1001,rec=1) u_new(:,:,:)
!close(1001)
!print*,"Finished saving ", fout

! save original u field
!fout2='/home/bgreene/fortran/u_orig.out'
!print*,"Saving file: ", fout2
!open(1002,file=TRIM(fout2),access='direct',status='unknown',recl=8*nx*ny*nz)
!    write(1002,rec=1) u(1:nx,:,:)
!close(1002)
!print*,"Finished saving ", fout2

! save interpolated v field
!fout3='/home/bgreene/fortran/v_interp.out'
!print*,"Saving file: ", fout3
!open(1003,file=TRIM(fout3),access='direct',status='unknown',recl=8*nx_new*ny_new*nz_new)
!    write(1003,rec=1) v_new(1:nx_new,1:ny_new,1:nz_new)
!close(1003)
!print*,"Finished saving ", fout3

! save original v field
!fout4='/home/bgreene/fortran/v_orig.out'
!print*,"Saving file: ", fout4
!open(1004,file=TRIM(fout4),access='direct',status='unknown',recl=8*nx*ny*nz)
!    write(1004,rec=1) v(1:nx,1:ny,1:nz)
!close(1004)
!print*,"Finished saving ", fout4

! save interpolated theta field
!fout5='/home/bgreene/fortran/theta_interp.out'
!print*,"Saving file: ", fout5
!open(1005,file=TRIM(fout5),access='direct',status='unknown',recl=8*nx_new*ny_new*nz_new)
!    write(1005,rec=1) theta_new(1:nx_new,1:ny_new,1:nz_new)
!close(1005)
!print*,"Finished saving ", fout5

! save original theta field
!fout6='/home/bgreene/fortran/theta_orig.out'
!print*,"Saving file: ", fout6
!open(1006,file=TRIM(fout6),access='direct',status='unknown',recl=8*nx*ny*(nz-1))
!    write(1006,rec=1) theta(1:nx,1:ny,1:nz-1)
!close(1006)
!print*,"Finished saving ", fout6

! now try saving new interpolated vel_sc.out so new simulation can use
fout7='/home/bgreene/simulations/F_256_interp/vel_sc.out'
print*,"Saving file: ", fout7
open(1007,file=TRIM(fout7),access='stream',status='unknown')
write(1007) u_new_ld(:,:,1:nz_new+1), v_new_ld(:,:,1:nz_new+1), &
                  w_new_ld(:,:,1:nz_new+1), theta_new_ld(:,:,1:nz_new+1), &
                  RHSx_new_ld(:,:,1:nz_new+1), RHSy_new_ld(:,:,1:nz_new+1), &
                  RHSz_new_ld(:,:,1:nz_new+1), RHS_T_new_ld(:,:,1:nz_new+1), &
                  sgs_t3_new(:,:,1), psi_m_new, Cs_opt2_new_ld, &
                  F_LM_new_ld, F_MM_new_ld, F_QN_new_ld, F_NN_new_ld, &
                  T_s_init_new
close(1007)
print*,"Finished saving ", fout7

END PROGRAM interp3d

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SUBROUTINE interpolate(nx, ny, nz, lx, ly, lz, T, nx_new, ny_new, nz_new,T_new)

IMPLICIT NONE

integer, parameter :: rprec = kind (1.d0)

! declare input params
INTEGER :: nx, ny, nz, nx_new, ny_new, nz_new
REAL(KIND=rprec) :: lx, ly, lz
REAL(KIND=rprec),DIMENSION(nx,ny,nz) :: T
REAL(KIND=rprec),DIMENSION(nx_new,ny_new,nz_new) :: T_new
! declare params to be used
INTEGER :: i, j, k, ii, jj, kk, i_x0, i_x1, j_y0, j_y1, k_z0, k_z1 ! counters and indices
REAL(KIND=rprec),DIMENSION(nx) :: xx ! linspace of x values
REAL(KIND=rprec),DIMENSION(ny) :: yy ! linspace of y values
REAL(KIND=rprec),DIMENSION(nz) :: zz ! linspace of z values
REAL(KIND=rprec),DIMENSION(nx_new) :: xx_new ! linspace of x_new values
REAL(KIND=rprec),DIMENSION(ny_new) :: yy_new ! linspace of y_new values
REAL(KIND=rprec),DIMENSION(nz_new) :: zz_new ! linspace of z_new values
REAL(KIND=rprec) :: xd, yd, zd, T00, T01, T10, T11, T0, T1 ! intermediate interpolated vals

! create linspace for xx, yy, zz
do i=1, nx
    xx(i) = 0. + lx * (i-1) / (nx-1)
end do

do i=1, ny
    yy(i) = 0. + ly * (i-1) / (ny-1)
end do

do i=1, nz
    zz(i) = 0. + lz * (i-1) / (nz-1)
end do

! create linspaces for new xx,yy,zz
do i=1, nx_new
    xx_new(i) = 0. + lx * (i-1) / (nx_new-1)
end do

do i=1, ny_new
    yy_new(i) = 0. + ly * (i-1) / (ny_new-1)
end do

do i=1, nz_new
    zz_new(i) = 0. + lz * (i-1) / (nz_new-1)
end do

! begin big loop
do i=1, nx_new
    do j=1, ny_new
        do k=1, nz_new
            ! find nearest x, y, z vertices of old grid xx, yy, zz
            ! loop through xx and compare to xx_new(i)
            do ii=1, nx
                if (xx_new(i) - xx(ii) .ge. 0.) then
                    i_x0 = ii
                else if (xx_new(i) - xx(ii) .le. 0.) then
                    exit
                end if
            end do
            i_x1 = i_x0 + 1
            if (i_x1 .ge. nx) then
                i_x0 = nx - 1
                i_x1 = nx
            end if
            ! now calculate xd
            xd = (xx_new(i) - xx(i_x0)) / (xx(i_x1) - xx(i_x0))
            
            ! now do the same thing for y
            do jj=1, ny
                if (yy_new(j) - yy(jj) .ge. 0.) then
                    j_y0 = jj
                else if (yy_new(j) - yy(jj) .le. 0.) then
                    exit
                end if
            end do
            j_y1 = j_y0 + 1
            if (j_y1 .ge. ny) then
                j_y0 = ny - 1
                j_y1 = ny
            end if
            ! now calculate yd
            yd = (yy_new(j) - yy(j_y0)) / (yy(j_y1) - yy(j_y0))    
            
            ! same thing for z
            do kk=1, nz
                if (zz_new(k) - zz(kk) .ge. 0.) then
                    k_z0 = kk
                else if (zz_new(k) - zz(kk) .le. 0.) then
                    exit
                end if
            end do
            k_z1 = k_z0 + 1
            if (k_z1 .ge. nz) then
                k_z0 = nz - 1
                k_z1 = nz
            end if
            ! now calculate zd
            zd = (zz_new(k) - zz(k_z0)) / (zz(k_z1) - zz(k_z0))            
            
            ! interpolate along x
            T00 = (T(i_x0,j_y0,k_z0) * (1.-xd)) + (T(i_x1,j_y0,k_z0) * xd)
            T01 = (T(i_x0,j_y0,k_z1) * (1.-xd)) + (T(i_x1,j_y0,k_z1) * xd)
            T10 = (T(i_x0,j_y1,k_z0) * (1.-xd)) + (T(i_x1,j_y1,k_z0) * xd)
            T11 = (T(i_x0,j_y1,k_z1) * (1.-xd)) + (T(i_x1,j_y1,k_z1) * xd)
            ! interpolate along y
            T0 = (T00 * (1.-yd)) + (T10 * yd)
            T1 = (T01 * (1.-yd)) + (T11 * yd)
            ! interpolate along z
            T_new(i,j,k) = (T0 * (1.-zd)) + (T1 * zd)
        end do
    end do
end do

END SUBROUTINE interpolate

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SUBROUTINE interpolate2d(nx, ny, lx, ly, T, nx_new, ny_new, T_new)

IMPLICIT NONE

! declare input params
INTEGER :: nx, ny, nx_new, ny_new
REAL(KIND=8) :: lx, ly
REAL(KIND=8),DIMENSION(nx,ny) :: T
REAL(KIND=8),DIMENSION(nx_new,ny_new) :: T_new
! declare params to be used
INTEGER :: i, j, ii, jj, i_x0, i_x1, j_y0, j_y1 ! counters and indices
REAL(KIND=8),DIMENSION(nx) :: xx ! linspace of x values
REAL(KIND=8),DIMENSION(ny) :: yy ! linspace of y values
REAL(KIND=8),DIMENSION(nx_new) :: xx_new ! linspace of x_new values
REAL(KIND=8),DIMENSION(ny_new) :: yy_new ! linspace of y_new values
REAL(KIND=8) :: T00, T01, T10, T11 ! intermediate interpolated vals

! create linspace for xx, yy
do i=1, nx
    xx(i) = 0. + lx * (i-1) / (nx-1)
end do

do i=1, ny
    yy(i) = 0. + ly * (i-1) / (ny-1)
end do

! create linspaces for new xx, yy
do i=1, nx_new
    xx_new(i) = 0. + lx * (i-1) / (nx_new-1)
end do

do i=1, ny_new
    yy_new(i) = 0. + ly * (i-1) / (ny_new-1)
end do

! begin big loop
do i=1, nx_new
    do j=1, ny_new
        ! find nearest x, y, z vertices of old grid xx, yy
        ! loop through xx and compare to xx_new(i)
        do ii=1, nx
            if (xx_new(i) - xx(ii) .ge. 0.) then
                i_x0 = ii
            else if (xx_new(i) - xx(ii) .le. 0.) then
                exit
            end if
        end do
        i_x1 = i_x0 + 1
        if (i_x1 .ge. nx) then
            i_x0 = nx - 1
            i_x1 = nx
        end if

        ! now do the same thing for y
        do jj=1, ny
            if (yy_new(j) - yy(jj) .ge. 0.) then
                j_y0 = jj
            else if (yy_new(j) - yy(jj) .le. 0.) then
                exit
            end if
        end do
        j_y1 = j_y0 + 1
        if (j_y1 .ge. ny) then
            j_y0 = ny - 1
            j_y1 = ny
        end if

        ! calculate the 4 contributing terms
        T00 = (((xx(i_x1) - xx_new(i)) * (yy(j_y1) - yy_new(j))) / &
              ((xx(i_x1) - xx(i_x0)) * (yy(j_y1) - yy(j_y0))))
        T10 = (((xx_new(i) - xx(i_x0)) * (yy(j_y1) - yy_new(j))) / &
              ((xx(i_x1) - xx(i_x0)) * (yy(j_y1) - yy(j_y0))))
        T01 = (((xx(i_x1) - xx_new(i)) * (yy_new(j) - yy(j_y0))) / &
              ((xx(i_x1) - xx(i_x0)) * (yy(j_y1) - yy(j_y0))))   
        T11 = (((xx_new(i) - xx(i_x0)) * (yy_new(j) - yy(j_y0))) / &
              ((xx(i_x1) - xx(i_x0)) * (yy(j_y1) - yy(j_y0))))
        ! combine
        T_new(i,j) = T00*T(i_x0,j_y0) + T10*T(i_x1,j_y0) + &
                     T01*T(i_x0,j_y1) + T11*T(i_x1,j_y1)
        
    end do
end do

END SUBROUTINE interpolate2d