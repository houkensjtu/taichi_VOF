      subroutine tilde
*     ----- begin comdeck common1    -----
      parameter (ibar2=22, jbar2=10, nprts=1, meshx=1, meshy=1, nvor=10)
      parameter (mshx=meshx+1, mshy=meshy+1)
c
      real nu, normx, normy
      integer cycle, wl, wr, wt, wb
c
      common /fv/ acom(1), un(ibar2,jbar2), vn(ibar2,jbar2), pn(ibar2
     1 ,jbar2), fn(ibar2,jbar2), u(ibar2,jbar2), v(ibar2,jbar2), p(ibar2
     2 ,jbar2), f(ibar2,jbar2), peta(ibar2,jbar2), beta(ibar2,jbar2), nf
     3 (ibar2,jbar2), tanth(ibar2,jbar2), ps(ibar2,jbar2)
c
      common /me/ x(ibar2), xi(ibar2), rxi(ibar2), delx(ibar2), rdx
     1 (ibar2), rx(ibar2), y(jbar2), yj(jbar2), ryj(jbar2), dely(jbar2),
     2 rdy(jbar2), xl(mshx), xc(meshx), dxmn(meshx), nxl(meshx), nxr
     3 (meshx), yl(mshy), yc(meshy), dymn(meshy), nyl(meshy), nyr(meshy)
      common /pv/ xp(nprts), yp(nprts), ip(nprts), jp(nprts), nr(nvor),
     1 pr(nvor), vol(nvor), name(10)
c
      common /iv/ ibar, jbar, imax, jmax, im1, jm1, im2, jm2, nkx, nky,
     1 cycle, delt, t, autot, prtdt, twprt, pltdt, twplt, twfin, flht,
     2 nu, csq, rcsq, nmat, rhof, rhofc, rhod, nvrm, nreg,vchgt,rdtexp,
     3 isurf10, sigma, cangle, tanca, icyl, cyl, gx, gy, ui, vi, omg,
     4 alpha, wl, wr, wb, wt, np, iter, epsi, flg, flgc, fnoc, nocon,
     5 nflgc, isymplt, imovy, velmx, velmx1, xshft, yshft, xmin, xmax,
     6 ymin, ymax, sf, xpl, xpr, ypb, ypt, npx, npy, ipl, ipr, jpb, jpt,
     7 dtvis,dtsft
c
      common /const/ emf, emf1, em6, em10, ep10, pi, rpd
c
      common /last/ zcom
*     ----- end comdeck common1    -----
c
c     *** compute temporary u and v explicitly
c
      do 20 j=2,jm1
      do 20 i=2,im1
      u(i,j)=0.0
      rdelx=1.0/(delx(i)+delx(i+1))
      rdely=1.0/(dely(j)+dely(j+1))
      if (f(i,j)+f(i+1,j).lt.emf.and.nmat.eq.1) go to 10
      if (beta(i,j).lt.0.0.or.beta(i+1,j).lt.0.0) go to 10
      sgu=sign(1.0,un(i,j))
      dudr=(un(i+1,j)-un(i,j))*rdx(i+1)
      dudl=(un(i,j)-un(i-1,j))*rdx(i)
      rdxa=delx(i)+delx(i+1)+alpha*sgu*(delx(i+1)-delx(i))
      rdxa=1.0/rdxa
      fux=rdxa*un(i,j)*(delx(i)*dudr+delx(i+1)*dudl+alpha*sgu*(delx(i+1)
     1 *dudl-delx(i)*dudr))
      vbt=(delx(i)*vn(i+1,j)+delx(i+1)*vn(i,j))*rdelx
      vbb=(delx(i)*vn(i+1,j-1)+delx(i+1)*vn(i,j-1))*rdelx
      vav=0.5*(vbt+vbb)
      dyt=0.5*(dely(j)+dely(j+1))
      dyb=0.5*(dely(j-1)+dely(j))
      dudt=(un(i,j+1)-un(i,j))/dyt
      dudb=(un(i,j)-un(i,j-1))/dyb
      sgv=sign(1.0,vav)
      dya=dyt+dyb+alpha*sgv*(dyt-dyb)
      fuy=(vav/dya)*(dyb*dudt+dyt*dudb+alpha*sgv*(dyt*dudb-dyb*dudt))
      ubdyt=(dely(j)*un(i,j+1)+dely(j+1)*un(i,j))/(dely(j)+dely(j+1))
      ubdyb=(dely(j-1)*un(i,j)+dely(j)*un(i,j-1))/(dely(j)+dely(j-1))
      dudxsq=2.0*(un(i-1,j)*rdx(i)/(delx(i)+delx(i+1))+un(i+1,j)*rdx(i+1
     1 )/(delx(i)+delx(i+1))-un(i,j)*rdx(i)*rdx(i+1))
      dudyt=(un(i,j+1)*dely(j)*rdy(j+1)-un(i,j)*dely(j+1)*rdy(j)-ubdyt*
     1 (dely(j)*rdy(j+1)-dely(j+1)*rdy(j)))/(0.5*(dely(j)+dely(j+1)))
      dudyb=(un(i,j)*dely(j-1)*rdy(j)-un(i,j-1)*dely(j)*rdy(j-1)-ubdyb*
     1 (dely(j-1)*rdy(j)-dely(j)*rdy(j-1)))/(0.5*(dely(j-1)+dely(j)))
      dudysq=(dudyt-dudyb)*rdy(j)
      dudxl=(un(i,j)-un(i-1,j))*rdx(i)
      dudxr=(un(i+1,j)-un(i,j))*rdx(i+1)
      rxdudx=rx(i)*(delx(i+1)*dudxl+delx(i)*dudxr)/(delx(i)+delx(i+1))
      rxsqu=un(i,j)*rx(i)**2
      visx=nu*(dudxsq+dudysq+cyl*rxdudx-cyl*rxsqu)
      rhox=(rhofc+rhod*f(i,j))*delx(i+1)+(rhofc+rhod*f(i+1,j))*delx(i)
      u(i,j)=un(i,j)+delt*((p(i,j)-p(i+1,j))*2.0/rhox+gx-fux-fuy+visx)
   10 continue
      v(i,j)=0.0
      if (f(i,j)+f(i,j+1).lt.emf.and.nmat.eq.1) go to 20
      if (beta(i,j).lt.0.0.or.beta(i,j+1).lt.0.0) go to 20
      ubr=(dely(j+1)*un(i,j)+dely(j)*un(i,j+1))*rdely
      ubl=(dely(j+1)*un(i-1,j)+dely(j)*un(i-1,j+1))*rdely
      uav=0.5*(ubr+ubl)
      dxr=0.5*(delx(i)+delx(i+1))
      dxl=0.5*(delx(i)+delx(i-1))
      sgu=sign(1.0,uav)
      dxa=dxr+dxl+alpha*sgu*(dxr-dxl)
      dvdr=(vn(i+1,j)-vn(i,j))/dxr
      dvdl=(vn(i,j)-vn(i-1,j))/dxl
      fvx=(uav/dxa)*(dxl*dvdr+dxr*dvdl+alpha*sgu*(dxr*dvdl-dxl*dvdr))
      sgv=sign(1.0,vn(i,j))
      dya=dely(j+1)+dely(j)+alpha*sgv*(dely(j+1)-dely(j))
      dvdt=(vn(i,j+1)-vn(i,j))*rdy(j+1)
      dvdb=(vn(i,j)-vn(i,j-1))*rdy(j)
      fvy=(vn(i,j)/dya)*(dely(j)*dvdt+dely(j+1)*dvdb+alpha*sgv*(dely(j+1
     1 )*dvdb-dely(j)*dvdt))
      vbdyr=(delx(i+1)*vn(i,j)+delx(i)*vn(i+1,j))/(delx(i)+delx(i+1))
      vbdyl=(delx(i)*vn(i-1,j)+delx(i-1)*vn(i,j))/(delx(i)+delx(i-1))
      dvdxr=(vn(i+1,j)*delx(i)*rdx(i+1)-vn(i,j)*delx(i+1)*rdx(i)-vbdyr*
     1 (delx(i)*rdx(i+1)-delx(i+1)*rdx(i)))/(0.5*(delx(i+1)+delx(i)))
      dvdxl=(vn(i,j)*delx(i-1)*rdx(i)-vn(i-1,j)*delx(i)*rdx(i-1)-vbdyl*
     1 (delx(i-1)*rdx(i)-delx(i)*rdx(i-1)))/(0.5*(delx(i)+delx(i-1)))
      dvdxsq=(dvdxr-dvdxl)*rdx(i)
      dvdysq=2.0*(vn(i,j-1)*rdy(j)/(dely(j+1)+dely(j))-vn(i,j)*rdy(j+1)
     1 *rdy(j)+vn(i,j+1)*rdy(j+1)/(dely(j+1)+dely(j)))
      dvdxrx=(vbdyr-vbdyl)*rdx(i)*rxi(i)
      visy=nu*(dvdxsq+dvdysq+cyl*dvdxrx)
      rhoy=(rhofc+rhod*f(i,j))*dely(j+1)+(rhofc+rhod*f(i,j+1))*dely(j)
      v(i,j)=vn(i,j)+delt*((p(i,j)-p(i,j+1))*2.0/rhoy+gy-fvx-fvy+visy)
   20 continue
      return
      end
