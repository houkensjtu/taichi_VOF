      subroutine pressit
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
c     *** pressure iteration
c
c     *** test for convergence
c
   10 if (flg.eq.0.) go to 140
      iter=iter+1
      itmax=1000
      if (iter.lt.itmax) go to 20
      fnoc=1.0
      nocon=nocon+1
      go to 140
   20 flg=0.0
c
c     *** compute updated cell pressure and velocities
c
      do 130 j=jpb,jpt
      do 130 i=ipl,ipr
      if (beta(i,j).lt.0.0) go to 130
      if (nmat.eq.2) go to 80
      if (f(i,j).lt.emf) go to 130
      if (nf(i,j).eq.0) go to 80
c
c     *** calculate pressure for surface cells
c
      nff=nf(i,j)
      l=i
      m=j
      go to (30,40,50,60,130), nff
   30 l=i-1
      go to 70
   40 l=i+1
      go to 70
   50 m=j-1
      go to 70
   60 m=j+1
   70 continue
      nfel=nf(i-1,j)
      nfer=nf(i+1,j)
      nfeb=nf(i,j-1)
      nfet=nf(i,j+1)
      nfe=max0(nfel,nfer,nfeb,nfet)
      psurf=ps(i,j)+pr(nfe)
      plm=p(l,m)
      if (nf(l,m).ne.0.and.beta(i,j).gt.0.0) plm=psurf
      delp=(1.0-peta(i,j))*plm+peta(i,j)*psurf-p(i,j)
      go to 90
   80 continue
      dij=rdx(i)*(u(i,j)-u(i-1,j))+rdy(j)*(v(i,j)-v(i,j-1))+cyl*0.5*rxi
     1 (i)*(u(i,j)+u(i-1,j))
      rhor=rhof/(rhofc+rhod*f(i,j))
      dfun=dij+rhor*rcsq*(p(i,j)-pn(i,j))/delt
c
c     *** set flag indicating convergence
c
      if (abs(dfun).ge.epsi) flg=1.0
      delp=-beta(i,j)*dfun*peta(i,j)
   90 continue
      p(i,j)=p(i,j)+delp
      ctos=delt*rdtexp
      comg=amin1(ctos**2,1.0)
      dptc=2.0*delt*delp*comg
      if (beta(i+1,j).lt.0.0) go to 100
      rhoxr=(rhofc+rhod*f(i,j))*delx(i+1)+(rhofc+rhod*f(i+1,j))*delx(i)
      u(i,j)=u(i,j)+dptc/rhoxr
  100 if (beta(i-1,j).lt.0.0) go to 110
      rhoxl=(rhofc+rhod*f(i-1,j))*delx(i)+(rhofc+rhod*f(i,j))*delx(i-1)
      u(i-1,j)=u(i-1,j)-dptc/rhoxl
  110 if (beta(i,j+1).lt.0.0) go to 120
      rhoyt=(rhofc+rhod*f(i,j))*dely(j+1)+(rhofc+rhod*f(i,j+1))*dely(j)
      v(i,j)=v(i,j)+dptc/rhoyt
  120 if (beta(i,j-1).lt.0.0) go to 130
      rhoyb=(rhofc+rhod*f(i,j-1))*dely(j)+(rhofc+rhod*f(i,j))*dely(j-1)
      v(i,j-1)=v(i,j-1)-dptc/rhoyb
  130 continue
      call bc
      go to 10
  140 continue
      return
      end
