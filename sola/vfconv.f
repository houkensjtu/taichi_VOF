      subroutine vfconv
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
c     *** convect the volume of fluid function f
c
      if (cycle.lt.1) go to 40
      flgc=0.0
      do 30 j=1,jm1
      do 30 i=1,im1
      vx=u(i,j)*delt
      vy=v(i,j)*delt
      abvx=abs(vx)
      abvy=abs(vy)
      if (abvx.gt.0.5*delx(i).or.abvy.gt.0.5*dely(j)) flgc=1.0
      ia=i+1
      id=i
      idm=max0(i-1,1)
      rb=x(i)
      ra=xi(i+1)
      rd=xi(i)
      if (vx.ge.0.0) go to 10
      ia=i
      id=i+1
      idm=min0(i+2,imax)
      ra=xi(i)
      rd=xi(i+1)
   10 continue
      iad=ia
      if (nf(id,j).eq.3.or.nf(id,j).eq.4) iad=id
      if (fn(ia,j).lt.emf.or.fn(idm,j).lt.emf) iad=ia
      fdm=amax1(fn(idm,j),fn(id,j))
      fx1=fn(iad,j)*abs(vx)+amax1((fdm-fn(iad,j))*abs(vx)-(fdm-fn(id,j))
     1 *delx(id),0.0)
      fx=amin1(fx1,fn(id,j)*delx(id))
      f(id,j)=f(id,j)-fx*rdx(id)*((abs(rb/rd))*cyl+(1.0-cyl))
      f(ia,j)=f(ia,j)+fx*rdx(ia)*((abs(rb/ra))*cyl+(1.0-cyl))
      ja=j+1
      jd=j
      jdm=max0(j-1,1)
      if (vy.ge.0.0) go to 20
      ja=j
      jd=j+1
      jdm=min0(j+2,jmax)
   20 continue
      jad=ja
      if (nf(i,jd).eq.1.or.nf(i,jd).eq.2) jad=jd
      if (fn(i,ja).lt.emf.or.fn(i,jdm).lt.emf) jad=ja
      fdm=amax1(fn(i,jdm),fn(i,jd))
      fy1=fn(i,jad)*abs(vy)+amax1((fdm-fn(i,jad))*abs(vy)-(fdm-fn(i,jd))
     1 *dely(jd),0.0)
      fy=amin1(fy1,fn(i,jd)*dely(jd))
      f(i,jd)=f(i,jd)-fy*rdy(jd)
      f(i,ja)=f(i,ja)+fy*rdy(ja)
   30 continue
   40 continue
      do 80 j=2,jm1
      do 80 i=2,im1
      if (beta(i,j).lt.0.0) go to 80
      vchg=0.0
      if (f(i,j).gt.emf.and.f(i,j).lt.emf1) go to 60
      if (f(i,j).ge.emf1) go to 50
      vchg=f(i,j)
      f(i,j)=0.0
      go to 60
   50 continue
      vchg=-(1.0-f(i,j))
      f(i,j)=1.0
   60 continue
      vchgt=vchgt+vchg*delx(i)*dely(j)*(xi(i)*2.0*pi*cyl+(1.0-cyl))
      if (f(i,j).lt.emf1) go to 80
      if (f(i+1,j).lt.emf) go to 70
      if (f(i-1,j).lt.emf) go to 70
      if (f(i,j+1).lt.emf) go to 70
      if (f(i,j-1).lt.emf) go to 70
      go to 80
   70 f(i,j)=f(i,j)-1.1*emf
      vchg=1.1*emf
      vchgt=vchgt+vchg*delx(i)*dely(j)*(xi(i)*2.0*pi*cyl+(1.0-cyl))
   80 continue
c
c     *** special boundary conditions for f
c
      return
      end
