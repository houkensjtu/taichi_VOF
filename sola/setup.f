      subroutine setup
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
c     *** compute constant terms and initialize necessary variables
c
c     *** set parameter statement value into constant
c
      nvrm=nvor
c
      cyl=float(icyl)
      emf1=1.0-emf
      t=0.0
      iter=0
      cycle=0
      twprt=0.0
      twplt=0.0
      vchgt=0.0
      nocon=0
      nflgc=0
      fnoc=0.0
      rcsq=1.0/(rhof*csq)
      if (csq.lt.0.0) rcsq=0.0
      if (nmat.eq.1) rhofc=rhof
      rhod=rhof-rhofc
      if (cangle.eq.90.0) cangle=cangle-em6
      cangle=cangle*rpd
      tanca=tan(cangle)
      ipl=2
      if (wl.eq.5) ipl=3
      ipr=im1
      if (wr.eq.5) ipr=im2
      jpb=2
      if (wb.eq.5) jpb=3
      jpt=jm1
      if (wt.eq.5) jpt=jm2
c
c     *** set constant terms for plotting
c
      xmin=x(1)
      xmax=x(im1)
      if (isymplt.gt.0) xmin=-xmax
      ymin=y(1)
      ymax=y(jm1)
      d1=xmax-xmin
      d2=ymax-ymin
      d3=amax1(d1,d2)
      sf=1.0/d3
      xshft=0.5*(1.0-d1*sf)
      yshft=0.5*(1.0-d2*sf)
      dxmin=ep10
      do 10 i=2,im1
   10 dxmin=amin1(delx(i),dxmin)
      dymin=ep10
      do 20 i=2,jm1
   20 dymin=amin1(dely(i),dymin)
      velmx1=amin1(dxmin,dymin)/velmx
c
c     *** determine sloped boundary location
c
c     *** compute initial volume fraction function f in cells
c
      do 40 i=1,imax
      do 30 j=2,jmax
      f(i,j)=1.0
      if (flht.gt.y(j-1).and.flht.lt.y(j)) f(i,j)=rdy(j)*(flht-y(j-1))
      if (y(j-1).ge.flht) f(i,j)=0.0
   30 continue
      f(i,1)=f(i,2)
   40 continue
c
c     *** generate special f-function (fluid) configuration
c
c     *** calculate dtvis and dtsft
c
      ds=1.0e+10
      dtvis=1.0e+10
      dtsft=1.0e+10
      do 50 i=2,im1
      do 50 j=2,jm1
      dxsq=delx(i)**2
      dysq=dely(j)**2
      rdsq=dxsq*dysq/(dxsq+dysq)
      rdsq=rdsq/(3.0*nu+1.0e-10)
      dtvis=amin1(dtvis,rdsq)
      ds=amin1(delx(i),dely(j),ds)
   50 continue
      sigx=sigma
      rhomn=amin1(rhof,rhofc)
      if(sigx.eq.0.0) sigx=em10
      dtm=sqrt(rhomn*ds**3/(sigx*4.0*(1.0+cyl)))
      dtsft=amin1(dtsft,dtm)
c
c     *** calculate beta(i,j) for mesh
c
      rdtexp= 2.0*sqrt(abs(csq))/ds
      if(csq.lt.0.0) rdtexp= 1.0e+10
      ctos=delt*rdtexp
      comg= amin1(ctos**2,1.0)
      omg1=(omg-1.0)*comg+1.0
      do 55 i=2,im1
      do 55 j=2,jm1
      rhxr=(rhofc+rhod*f(i,j))*delx(i+1)+(rhofc+rhod*f(i+1,j))*delx(i)
      rhxl=(rhofc+rhod*f(i-1,j))*delx(i)+(rhofc+rhod*f(i,j))*delx(i-1)
      rhyt=(rhofc+rhod*f(i,j))*dely(j+1)+(rhofc+rhod*f(i,j+1))*dely(j)
      rhyb=(rhofc+rhod*f(i,j-1))*dely(j)+(rhofc+rhod*f(i,j))*dely(j-1)
      xx=delt*rdx(i)*(2.0/rhxl+2.0/rhxr)+delt*rdy(j)*(2.0/rhyt+2.0/rhyb)
      rhor=rhof/(rhofc+rhod*f(i,j))
      beta(i,j)=omg1/(xx*comg+rcsq*rhor/delt)
   55 continue
c
c     *** set beta(i,j)= -1.0 in obstacle cells
c         must be done by hand in general
c
c     *** print beta(i,j) on film and paper
c
      if (imovy.eq.1) go to 70
      write (12,210)
      do 60 j=1,jm1
      do 60 i=1,im1
      write (12,220) i,j,beta(i,j)
   60 continue
   70 continue
      write (6,210)
      do 80 j=1,jm1
      do 80 i=1,im1
      write (6,220) i,j,beta(i,j)
   80 continue
c
c     *** calculate hydrostatic pressure
c
      do 90 i=2,im1
      p(i,jmax)=0.0
      do 90 j=2,jm1
      jj=jm1-j+2
      rhoya=(rhofc+rhod*f(i,jj))*dely(jj)*0.5+(rhofc+rhod*f(i,jj+1))
     1 *dely(jj+1)*0.5
      if (nmat.eq.1) rhoya=(amin1(f(i,jj+1),0.5)*dely(jj+1)+amax1(0.0,f
     1 (i,jj)-0.5)*dely(jj))*rhof
      p(i,jj)=p(i,jj+1)-gy*rhoya
   90 continue
c
c     *** particle set up
c
      np=npy*(1+npx)
      if (np.eq.0) go to 160
      dxp=(xpr-xpl)/float(npx)
      dyp=(ypt-ypb)/float(npy)
      k=0
      do 100 jn=1,npy,2
      do 100 in=1,npx
      k=k+1
      xp(k)=xpl+(float(in)-0.5)*dxp
      yp(k)=ypb+(float(jn)-1.0)*dyp
      if (yp(k).gt.ypt) yp(k)=ypt
  100 continue
      do 110 jn=2,npy,2
      k=k+1
      xp(k)=xpl
      yp(k)=ypb+(float(jn)-1.0)*dyp
      if (yp(k).gt.ypt) yp(k)=ypt
      do 110 in=1,npx
      k=k+1
      xp(k)=xpl+float(in)*dxp
      yp(k)=ypb+(float(jn)-1.0)*dyp
      if (yp(k).gt.ypt) yp(k)=ypt
  110 continue
      np=k
      do 150 k=1,np
      do 120 i=2,im1
      if (xp(k).ge.x(i-1).and.xp(k).le.x(i)) ip(k)=i
      if (x(i-1).gt.xpr) go to 130
  120 continue
  130 do 140 j=2,jm1
      if (yp(k).ge.y(j-1).and.yp(k).le.y(j)) jp(k)=j
      if (y(j-1).gt.ypt) go to 150
  140 continue
  150 continue
  160 continue
c
c     *** set initial surface pressure
c
      do 170 j=2,jm1
      do 170 i=2,im1
      ps(i,j)=0.0
  170 continue
c
c     *** set initial velocity field into u and v arrays
c
      do 180 i=2,im1
      do 180 j=2,jm1
      v(i,j)=vi
      u(i,j)=ui
      if (f(i,j).gt.emf.or.nmat.eq.2) go to 180
      u(i,j)=0.0
      v(i,j)=0.0
  180 continue
c
c     *** set initial void region quantities
c
      do 190 k=1,nvrm
      nr(k)=0
      pr(k)=0.0
  190 vol(k)=0.0
  200 return
c
  210 format (1h1)
  220 format (2x,5hbeta(,i2,1h,,i2,2h)=,1pe14.7)
      end
