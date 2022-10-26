      subroutine bc
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
c     *** set boundary conditions
c
      do 100 j=1,jmax
      f(1,j)=f(2,j)
      f(imax,j)=f(im1,j)
      p(1,j)=p(2,j)
      p(imax,j)=p(im1,j)
      go to (10,20,30,40,30), wl
   10 u(1,j)=0.0
      v(1,j)=v(2,j)
      go to 50
   20 u(1,j)=0.0
      v(1,j)=-v(2,j)*delx(1)/delx(2)
      go to 50
   30 if (iter.gt.0) go to 50
      u(1,j)=u(2,j)*(x(2)*rx(1)*cyl+1.0-cyl)
      v(1,j)=v(2,j)
      go to 50
   40 u(1,j)=u(im2,j)
      v(1,j)=v(im2,j)
      f(1,j)=f(im2,j)
   50 go to (60,70,80,90,80), wr
   60 u(im1,j)=0.0
      v(imax,j)=v(im1,j)
      go to 100
   70 u(im1,j)=0.0
      v(imax,j)=-v(im1,j)*delx(imax)/delx(im1)
      go to 100
   80 if (iter.gt.0) go to 100
      u(im1,j)=u(im2,j)*(x(im2)*rx(im1)*cyl+1.0-cyl)
      v(imax,j)=v(im1,j)
      go to 100
   90 u(im1,j)=u(2,j)
      v(im1,j)=v(2,j)
      p(im1,j)=p(2,j)
      ps(im1,j)=ps(2,j)
      f(im1,j)=f(2,j)
      v(imax,j)=v(3,j)
      f(imax,j)=f(3,j)
  100 continue
      do 200 i=1,imax
      f(i,1)=f(i,2)
      f(i,jmax)=f(i,jm1)
      p(i,1)=p(i,2)
      p(i,jmax)=p(i,jm1)
      go to (110,120,130,140,130), wt
  110 v(i,jm1)=0.0
      u(i,jmax)=u(i,jm1)
      go to 150
  120 v(i,jm1)=0.0
      u(i,jmax)=-u(i,jm1)*dely(jmax)/dely(jm1)
      go to 150
  130 if (iter.gt.0) go to 150
      v(i,jm1)=v(i,jm2)
      u(i,jmax)=u(i,jm1)
      go to 150
  140 v(i,jm1)=v(i,2)
      u(i,jm1)=u(i,2)
      p(i,jm1)=p(i,2)
      ps(i,jm1)=ps(i,2)
      f(i,jm1)=f(i,2)
      u(i,jmax)=u(i,3)
      f(i,jmax)=f(i,3)
  150 go to (160,170,180,190,180), wb
  160 v(i,1)=0.0
      u(i,1)=u(i,2)
      go to 200
  170 v(i,1)=0.0
      u(i,1)=-u(i,2)*dely(1)/dely(2)
      go to 200
  180 if (iter.gt.0) go to 200
      v(i,1)=v(i,2)
      u(i,1)=u(i,2)
      go to 200
  190 v(i,1)=v(i,jm2)
      u(i,1)=u(i,jm2)
      f(i,1)=f(i,jm2)
  200 continue
c
c     *** free surface and sloped boundary conditions
c
      do 450 i=2,im1
      xrp=rdx(i)+0.5*rxi(i)
      rxrp=1./xrp
      xrm=rdx(i)-0.5*rxi(i)
      if (xrm.gt.0.0) go to 210
      rxrm=0.0
      go to 220
  210 continue
      rxrm=1./xrm
  220 continue
      do 450 j=2,jm1
      if (beta(i,j).gt.0.0) go to 230
      bmr=0.0
      bmt=0.0
      bml=0.0
      bmb=0.0
      f(i,j)=0.0
      p(i,j)=0.0
      if (beta(i+1,j).gt.0.0) bmr=1.0
      if (beta(i,j+1).gt.0.0) bmt=1.0
      if (beta(i-1,j).gt.0.0) bml=1.0
      if (beta(i,j-1).gt.0.0) bmb=1.0
      bmtot=bmr+bmt+bml+bmb
      if (bmtot.le.0.0) go to 450
      f(i,j)=(bmr*f(i+1,j)+bmt*f(i,j+1)+bml*f(i-1,j)+bmb*f(i,j-1))/bmtot
      p(i,j)=(bmr*p(i+1,j)+bmt*p(i,j+1)+bml*p(i-1,j)+bmb*p(i,j-1))/bmtot
      go to 450
  230 continue
      if (nmat.eq.2) go to 450
      if (f(i,j).lt.emf.or.f(i,j).gt.emf1) go to 450
      nfsb=0
      if (f(i+1,j).lt.emf) nfsb=nfsb+1
      if (f(i,j+1).lt.emf) nfsb=nfsb+2
      if (f(i-1,j).lt.emf) nfsb=nfsb+4
      if (f(i,j-1).lt.emf) nfsb=nfsb+8
      if (nfsb.eq.0) go to 450
      if (nfsb.gt.8) go to 240
      go to (250,260,270,280,290,300,310,320), nfsb
  240 nfsb1=nfsb-8
      go to (330,340,350,360,370,380,390), nfsb1
  250 u(i,j)=(u(i-1,j)-delx(i)*rdy(j)*(v(i,j)-v(i,j-1)))*(1.0-cyl)+cyl*
     1 (u(i-1,j)*xrm*rxrp-rdy(j)*rxrp*(v(i,j)-v(i,j-1)))
      go to 410
  260 v(i,j)=(v(i,j-1)-dely(j)*rdx(i)*(u(i,j)-u(i-1,j)))*(1.0-cyl)+cyl*
     1 (v(i,j-1)-dely(j)*(xrp*u(i,j)-xrm*u(i-1,j)))
      go to 410
  270 u(i,j)=u(i-1,j)*(1.0-cyl)+cyl*u(i-1,j)
      go to 260
  280 u(i-1,j)=(u(i,j)+delx(i)*rdy(j)*(v(i,j)-v(i,j-1)))*(1.0-cyl)+cyl*
     1 (u(i,j)*xrp*rxrm+rdy(j)*rxrm*(v(i,j)-v(i,j-1)))
      go to 410
  290 u(i-1,j)=u(i-1,j-1)
      go to 250
  300 u(i-1,j)=u(i,j)*(1.0-cyl)+cyl*u(i,j)
      go to 260
  310 u(i-1,j)=u(i-1,j-1)
      u(i,j)=u(i,j-1)
      go to 260
  320 v(i,j-1)=(v(i,j)+dely(j)*rdx(i)*(u(i,j)-u(i-1,j)))*(1.0-cyl)+cyl*
     1 (v(i,j)+dely(j)*(xrp*u(i,j)-xrm*u(i-1,j)))
      go to 410
  330 u(i,j)=u(i-1,j)*(1.0-cyl)+cyl*u(i-1,j)
      go to 320
  340 v(i,j)=v(i-1,j)
      go to 320
  350 v(i,j)=v(i-1,j)
      v(i,j-1)=v(i-1,j-1)
      go to 250
  360 u(i-1,j)=u(i,j)*(1.0-cyl)+cyl*u(i,j)
      go to 320
  370 u(i,j)=u(i,j+1)
      u(i-1,j)=u(i-1,j+1)
      go to 320
  380 v(i,j)=v(i+1,j)
      v(i,j-1)=v(i+1,j-1)
      go to 280
  390 u(i,j)=u(i-1,j)*(1.0-cyl)+cyl*u(i-1,j)*xrm*rxrp
      v(i,j-1)=v(i,j)
      v(i,j+1)=v(i,j)
      go to 410
c
c     *** set velocities in empty cells adjacent to partial fluid cells
c
  410 continue
      if (flg.gt.0.5.and.iter.gt.0) go to 450
      if (f(i+1,j).gt.emf) go to 420
      if (f(i+1,j+1).lt.emf) v(i+1,j)=v(i,j)
      if (f(i+1,j-1).lt.emf) v(i+1,j-1)=v(i,j-1)
  420 if (f(i,j+1).gt.emf) go to 430
      if (f(i+1,j+1).lt.emf) u(i,j+1)=u(i,j)
      if (f(i-1,j+1).lt.emf) u(i-1,j+1)=u(i-1,j)
  430 if (f(i-1,j).gt.emf) go to 440
      if (f(i-1,j+1).lt.emf) v(i-1,j)=v(i,j)
      if (f(i-1,j-1).lt.emf) v(i-1,j-1)=v(i,j-1)
  440 if (f(i,j-1).gt.emf) go to 450
      if (f(i+1,j-1).lt.emf) u(i,j-1)=u(i,j)
      if (f(i-1,j-1).lt.emf) u(i-1,j-1)=u(i-1,j)
  450 continue
c
c     *** special velocity boundary conditions
c
      return
      end
