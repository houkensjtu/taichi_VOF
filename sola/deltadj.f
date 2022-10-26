      subroutine deltadj
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
c     *** delt (time step) adjustment
c
      deltn=delt
      if (flgc.lt.0.5) go to 20
      t=t-delt
      cycle=cycle-1
      delt=0.5*delt
      do 10 i=1,imax
      do 10 j=1,jmax
      p(i,j)=pn(i,j)
      f(i,j)=fn(i,j)
      u(i,j)=0.0
      v(i,j)=0.0
   10 continue
      nflgc=nflgc+1
   20 continue
      if (autot.lt.0.5.and.fnoc.lt.0.5) go to 35
      dumx=em10
      dvmx=em10
      if (fnoc.gt.0.5) delt=0.5*delt
      do 30 i=2,im1
      do 30 j=2,jm1
      udm=abs(un(i,j))/(xi(i+1)-xi(i))
      vdm=abs(vn(i,j))/(yj(j+1)-yj(j))
      dumx=amax1(dumx,udm)
      dvmx=amax1(dvmx,vdm)
   30 continue
      dtmp=1.01
      if (iter.gt.25) dtmp=0.99
      delto=delt*dtmp
      con=0.25
      delt=amin1(delto,con/dumx,con/dvmx,dtvis,dtsft)
      if (imovy.gt.0) delt=amin1(delt,pltdt)
   35 if(delt.eq.deltn .and. nmat.eq.1) go to 50
      ctos=delt*rdtexp
      comg=amin1(ctos**2,1.0)
      omg1=(omg-1.0)*comg+1.0
      do 40 i=1,imax
      do 40 j=1,jmax
      if (beta(i,j).lt.0.0) go to 40
      rhxr=(rhofc+rhod*f(i,j))*delx(i+1)+(rhofc+rhod*f(i+1,j))*delx(i)
      rhxl=(rhofc+rhod*f(i-1,j))*delx(i)+(rhofc+rhod*f(i,j))*delx(i-1)
      rhyt=(rhofc+rhod*f(i,j))*dely(j+1)+(rhofc+rhod*f(i,j+1))*dely(j)
      rhyb=(rhofc+rhod*f(i,j-1))*dely(j)+(rhofc+rhod*f(i,j))*dely(j-1)
      xx=delt*rdx(i)*(2.0/rhxl+2.0/rhxr)+delt*rdy(j)*(2.0/rhyt+2.0/rhyb)
      rhor=rhof/(rhofc+rhod*f(i,j))
      beta(i,j)=omg1/(xx*comg+rcsq*rhor/delt)
   40 continue
   50 continue
      return
      end
