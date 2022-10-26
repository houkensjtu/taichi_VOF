      subroutine petacal
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
c     *** determine the pressure interpolation factor peta
c     *** determine the surface tension pressure and
c     *** wall adhesion effects in surface cells
c
      do 10 i=1,imax
      do 10 j=1,jmax
      nf(i,j)=0
      ps(i,j)=0.0
   10 peta(i,j)=1.0
      ipass=0
      do 150 i=2,im1
      do 150 j=2,jm1
      tanth(i,j)=ep10
      if (beta(i,j).lt.0.0) go to 150
      if (f(i,j).lt.emf) nf(i,j)=6
      if (f(i,j).lt.emf.or.f(i,j).gt.emf1) go to 150
      if (f(i+1,j).lt.emf) go to 20
      if (f(i,j+1).lt.emf) go to 20
      if (f(i-1,j).lt.emf) go to 20
      if (f(i,j-1).lt.emf) go to 20
      go to 150
   20 continue
c
c     *** calculate the partial derivatives of f
c
      dxr=0.5*(delx(i)+delx(i+1))
      dxl=0.5*(delx(i)+delx(i-1))
      dyt=0.5*(dely(j)+dely(j+1))
      dyb=0.5*(dely(j)+dely(j-1))
      rxden=1.0/(dxr*dxl*(dxr+dxl))
      ryden=1.0/(dyt*dyb*(dyt+dyb))
      fl=f(i-1,j+1)
      if (beta(i-1,j+1).lt.0.0.or.(i.eq.2.and.wl.lt.3)) fl=1.0
      fc=f(i,j+1)
      if (beta(i,j+1).lt.0.0) fc=1.0
      fr=f(i+1,j+1)
      if (beta(i+1,j+1).lt.0.0.or.(i.eq.im1.and.wr.lt.3)) fr=1.0
      avft=fl*delx(i-1)+fc*delx(i)+fr*delx(i+1)
      fl=f(i-1,j-1)
      if (beta(i-1,j-1).lt.0.0.or.(i.eq.2.and.wl.lt.3)) fl=1.0
      fc=f(i,j-1)
      if (beta(i,j-1).lt.0.0) fc=1.0
      fr=f(i+1,j-1)
      if (beta(i+1,j-1).lt.0.0.or.(i.eq.im1.and.wr.lt.3)) fr=1.0
      avfb=fl*delx(i-1)+fc*delx(i)+fr*delx(i+1)
      fl=f(i-1,j)
      if (beta(i-1,j).lt.0.0.or.(i.eq.2.and.wl.lt.3)) fl=1.0
      fr=f(i+1,j)
      if (beta(i+1,j).lt.0.0.or.(i.eq.im1.and.wr.lt.3)) fr=1.0
      avfcy=fl*delx(i-1)+f(i,j)*delx(i)+fr*delx(i+1)
      fb=f(i,j-1)
      if (beta(i,j-1).lt.0.0.or.(j.eq.2.and.wb.lt.3)) fb=1.0
      ft=f(i,j+1)
      if (beta(i,j+1).lt.0.0.or.(j.eq.jm1.and.wt.lt.3)) ft=1.0
      avfcx=fb*dely(j-1)+f(i,j)*dely(j)+ft*dely(j+1)
      fb=f(i-1,j-1)
      if (beta(i-1,j-1).lt.0.0.or.(j.eq.2.and.wb.lt.3)) fb=1.0
      fc=f(i-1,j)
      if (beta(i-1,j).lt.0.0) fc=1.0
      ft=f(i-1,j+1)
      if (beta(i-1,j+1).lt.0.0.or.(j.eq.jm1.and.wt.lt.3)) ft=1.0
      avfl=fb*dely(j-1)+fc*dely(j)+ft*dely(j+1)
      fb=f(i+1,j-1)
      if (beta(i+1,j-1).lt.0.0.or.(j.eq.2.and.wb.lt.3)) fb=1.0
      fc=f(i+1,j)
      if (beta(i+1,j).lt.0.0) fc=1.0
      ft=f(i+1,j+1)
      if (beta(i+1,j+1).lt.0.0.or.(j.eq.jm1.and.wt.lt.3)) ft=1.0
      avfr=fb*dely(j-1)+fc*dely(j)+ft*dely(j+1)
c
c     *** boundary conditions for wall adhesion
c
      if (isurf10.eq.0.or.cangle.eq.0.0) go to 60
      if (beta(i+1,j).ge.0.0.and.i.ne.im1) go to 30
      avfr=avfcx+0.5*(delx(i)+delx(i+1))/tanca
      if (f(i,j+1).lt.emf.and.f(i,j-1).ge.emf) avft=avfcy-0.5*(dely(j)
     1 +dely(j+1))*tanca
      if (f(i,j-1).lt.emf.and.f(i,j+1).ge.emf) avfb=avfcy-0.5*(dely(j)
     1 +dely(j-1))*tanca
   30 if (beta(i,j+1).ge.0.0.and.j.ne.jm1) go to 40
      avft=avfcy+0.5*(dely(j)+dely(j+1))/tanca
      if (f(i+1,j).lt.emf.and.f(i-1,j).ge.emf) avfr=avfcx-0.5*(delx(i)
     1 +delx(i+1))*tanca
      if (f(i-1,j).lt.emf.and.f(i+1,j).ge.emf) avfl=avfcx-0.5*(delx(i)
     1 +delx(i-1))*tanca
   40 if (beta(i,j-1).ge.0.0.and.j.ne.2) go to 50
      avfb=avfcy+0.5*(dely(j)+dely(j-1))/tanca
      if (f(i+1,j).lt.emf.and.f(i-1,j).ge.emf) avfr=avfcx-0.5*(delx(i)
     1 +delx(i+1))*tanca
      if (f(i-1,j).lt.emf.and.f(i+1,j).ge.emf) avfl=avfcx-0.5*(delx(i)
     1 +delx(i-1))*tanca
   50 if (beta(i-1,j).ge.0.0.and.i.ne.2) go to 60
      if (cyl.gt.0.5.and.x(1).eq.0.0) go to 60
      avfl=avfcx+0.5*(delx(i)+delx(i-1))/tanca
      if (f(i,j+1).lt.emf.and.f(i,j-1).ge.emf) avft=avfcy-0.5*(dely(j)
     1 +dely(j+1))*tanca
      if (f(i,j-1).lt.emf.and.f(i,j+1).ge.emf) avfb=avfcy-0.5*(dely(j)
     1 +dely(j-1))*tanca
   60 continue
      xthm=3.0*amax1(avft,avfcy,avfb)/(delx(i-1)+delx(i)+delx(i+1))
      ythm=3.0*amax1(avfl,avfcx,avfr)/(dely(j-1)+dely(j)+dely(j+1))
      pfx=rxden*((avfr-avfcx)*dxl**2+(avfcx-avfl)*dxr**2)
      pfy=ryden*((avft-avfcy)*dyb**2+(avfcy-avfb)*dyt**2)
      pf=pfx**2+pfy**2
      if (pf.gt.em10) go to 70
      nf(i,j)=5
      p(i,j)=0.25*(p(i+1,j)+p(i,j+1)+p(i-1,j)+p(i,j-1))
      go to 150
   70 continue
c
c     *** determine the pressure interpolation cell nf
c
      abpfx=abs(pfx)
      abpfy=abs(pfy)
      l=i
      m=j
      if (abpfy.ge.abpfx) go to 80
      dxdyr=dely(j)*rdx(i)
      pfmn=pfy
      nf(i,j)=2
      l=i+1
      dmx=delx(i)
      dmin=0.5*(dmx+delx(i+1))
      if (pfx.gt.0.0) go to 90
      nf(i,j)=1
      pfmn=-pfy
      l=i-1
      dmx=delx(i)
      dmin=0.5*(dmx+delx(i-1))
      go to 90
   80 continue
      dxdyr=delx(i)*rdy(j)
      pfmn=-pfx
      nf(i,j)=4
      m=j+1
      dmx=dely(j)
      dmin=0.5*(dmx+dely(j+1))
      if (pfy.gt.0.0) go to 90
      nf(i,j)=3
      pfmn=pfx
      m=j-1
      dmx=dely(j)
      dmin=0.5*(dmx+dely(j-1))
   90 continue
      tanth(i,j)=pfmn
      abtan=abs(tanth(i,j))
c
c     *** determine the curvature and surface pressure
c
      dfs=(0.5-f(i,j))*dmx
      if (f(i,j).lt.0.5*abtan*dxdyr) dfs=0.5*dmx*(1.0+dxdyr*abtan-sqrt(8
     1 .0*f(i,j)*dxdyr*abtan))
      if (isurf10.lt.1) go to 140
      nfc=nf(i,j)
      pxr=(avfr-avfcx)/dxr
      pxl=(avfcx-avfl)/dxl
      pyt=(avft-avfcy)/dyt
      pyb=(avfcy-avfb)/dyb
      ydfs=-dfs
      if(nfc.eq.2 .or. nfc.eq.4) ydfs=dfs
      if(nfc.gt.2) go to 100
      dxdn=dely(j)
      xinb=ydfs+0.5*tanth(i,j)*dxdn
      xint=2.0*ydfs-xinb
      gp1=pyt
      px1=pxl
      if(xint.gt.0.0) px1=pxr
      if(abs(px1).lt.abs(gp1)) gp1=sign(1.0,gp1)/(abs(px1)+em10)
      gp2=pyb
      px2=pxr
      if(xinb.lt.0.0) px2=pxl
      if(abs(px2).lt.abs(gp2)) gp2=sign(1.0,gp2)/(abs(px2)+em10)
      go to 110
  100 dxdn=delx(i)
      yinr=ydfs+0.5*tanth(i,j)*dxdn
      yinl=2.0*ydfs-yinr
      gp1=pxr
      py1=pyt
      if(yinr.lt.0.0) py1=pyb
      if(abs(py1).lt.abs(gp1)) gp1=sign(1.0,gp1)/(abs(py1)+em10)
      gp2=pxl
      py2=pyb
      if(yinl.gt.0.0) py2=pyt
      if(abs(py2).lt.abs(gp2)) gp2=sign(1.0,gp2)/(abs(py2)+em10)
  110 gp1d=1.0+gp1*gp1
      gp2d=1.0+gp2*gp2
      curvxy=(gp2/sqrt(gp2d)-gp1/sqrt(gp1d))/dxdn
      curvcyl=0.0
      if (cyl.lt.1.0) go to 120
      xlitlr=xi(i)
      if (nfc.eq.1) xlitlr=x(i-1)+f(i,j)*delx(i)
      if (nfc.eq.2) xlitlr=x(i)-f(i,j)*delx(i)
      rlitlr=amin1(1.0/xlitlr,rxi(2))
      trig=abs(sin(atan(abtan)))
      if (nfc.le.2) trig=abs(cos(atan(abtan)))
      curvcyl=-cyl*trig*sign(1.0,pfx)*rlitlr
  120 curv=curvxy+curvcyl
      ps(i,j)=sigma*curv
      if (xthm.lt.1.0.or.ythm.lt.1.0) ps(i,j)=0.0
  140 continue
c
c     *** calculate peta
c
      nfsb=0
      if (f(i+1,j).lt.emf.or.i.eq.im1.or.beta(i+1,j).lt.0.0) nfsb=nfsb+1
      if (f(i,j+1).lt.emf.or.beta(i,j+1).lt.0.0) nfsb=nfsb+2
      if (f(i-1,j).lt.emf.or.beta(i-1,j).lt.0.0) nfsb=nfsb+4
      if (f(i,j-1).lt.emf.or.beta(i,j-1).lt.0.0) nfsb=nfsb+8
      if (nfsb.eq.15) ps(i,j)=0.0
      if (nmat.eq.2) go to 150
      peta(i,j)=1.0/(1.0-dfs/dmin)
      if (l.eq.1.or.l.eq.imax) peta(i,j)=1.0
      if (m.eq.1.or.m.eq.jmax) peta(i,j)=1.0
      if (beta(l,m).lt.0.0) peta(i,j)=1.0
  150 continue
c
      call lavore
c
      call cavovo
c
c     if necessary, determine pressures pr for void regions nf
c
      if (nmat.eq.2) go to 300
c
c     *** set peta in adjacent full cell
c
      do 290 j=1,jmax
      do 290 i=1,imax
      nff=nf(i,j)
      if (nff.eq.0.or.beta(i,j).lt.0.0) go to 290
      if (nff.gt.5) go to 280
      l=i
      m=j
      go to (230,240,250,260,290), nff
  230 l=i-1
      dmx=delx(l)
      dmin=0.5*(dmx+delx(i))
      go to 270
  240 l=i+1
      dmx=delx(l)
      dmin=0.5*(dmx+delx(i))
      go to 270
  250 m=j-1
      dmx=dely(m)
      dmin=0.5*(dmx+dely(j))
      go to 270
  260 m=j+1
      dmx=dely(m)
      dmin=0.5*(dmx+dely(j))
  270 continue
      if (nf(l,m).gt.0) go to 290
      ctos=delt*rdtexp
      comg=amin1(ctos**2,1.0)
      bpd=1.0/peta(l,m)-beta(l,m)*(1.0-peta(i,j))
     1*delt/(dmin*dmx)*(comg/rhof)
      peta(l,m)= 1.0/bpd
      go to 290
  280 continue
      p(i,j)=pr(nff)
  290 continue
  300 continue
      return
      end
