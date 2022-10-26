      subroutine parmov
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
c     *** marker particle movement section
c
      npt=0
      npn=0
      k=1
      kn=1
      pflg=1.0
      iper=im1
      if(wr.eq.4) iper=im2
      jper=jm1
      if(wt.eq.4) jper=jm2
   10 if (np.eq.npt) go to 150
c
c     *** claculate u weighted velocity of particle
c
      i=ip(k)
      j=jp(k)
      if (yp(k).gt.yj(j)) go to 20
      hpx=x(i)-xp(k)
      hmx=delx(i)-hpx
      hpy=yj(j)-yp(k)
      normy=(dely(j)+dely(j-1))*0.5
      hmy=normy-hpy
      utop=(u(i-1,j)*hpx+u(i,j)*hmx)*rdx(i)
      ubot=(u(i-1,j-1)*hpx+u(i,j-1)*hmx)*rdx(i)
      upart=(utop*hmy+ubot*hpy)/normy
      go to 30
   20 hpx=x(i)-xp(k)
      hmx=delx(i)-hpx
      hpy=yj(j+1)-yp(k)
      normy=(dely(j+1)+dely(j))*0.5
      hmy=normy-hpy
      utop=(u(i-1,j+1)*hpx+u(i,j+1)*hmx)*rdx(i)
      ubot=(u(i-1,j)*hpx+u(i,j)*hmx)*rdx(i)
      upart=(utop*hmy+ubot*hpy)/normy
c
c     *** calculate v weighted velocity of particle
c
   30 if (xp(k).gt.xi(i)) go to 40
      normx=(delx(i)+delx(i-1))*0.5
      rnormx=1.0/normx
      hpx=xi(i)-xp(k)
      hmx=normx-hpx
      hpy=y(j)-yp(k)
      hmy=dely(j)-hpy
      vtop=(v(i-1,j)*hpx+v(i,j)*hmx)*rnormx
      vbot=(v(i-1,j-1)*hpx+v(i,j-1)*hmx)*rnormx
      vpart=(vtop*hmy+vbot*hpy)*rdy(j)
      go to 50
   40 normx=(delx(i)+delx(i+1))*0.5
      rnormx=1.0/normx
      hpx=xi(i+1)-xp(k)
      hmx=normx-hpx
      hpy=y(j)-yp(k)
      hmy=dely(j)-hpy
      vtop=(v(i,j)*hpx+v(i+1,j)*hmx)*rnormx
      vbot=(v(i,j-1)*hpx+v(i+1,j-1)*hmx)*rnormx
      vpart=(vtop*hmy+vbot*hpy)*rdy(j)
   50 xpart=xp(k)+upart*delt
      ypart=yp(k)+vpart*delt
      if (xpart.gt.x(i)) ip(kn)=ip(k)+1
      if (xpart.lt.x(i-1)) ip(kn)=ip(k)-1
      if (ypart.gt.y(j)) jp(kn)=jp(k)+1
      if (ypart.lt.y(j-1)) jp(kn)=jp(k)-1
      xp(kn)=xpart
      yp(kn)=ypart
      if(xp(kn).lt.x(1)) go to 90
      if(yp(kn).lt.y(1)) go to 100
      if(xp(kn).gt.x(iper)) go to 110
      if(yp(kn).gt.y(jper)) go to 120
      go to 130
   90 if(wl.le.2) go to 130
      if(wl.ne.4) go to 140
      xp(kn)=xp(kn)+x(im2)-x(1)
      ip(kn)=ip(kn)+im2-1
      go to 130
  100 if(wb.le.2) go to 130
      if(wb.ne.4) go to 140
      yp(kn)=yp(kn)+y(jm2)-y(1)
      jp(kn)=jp(kn)+jm2-1
      go to 130
  110 if(wr.le.2) go to 130
      if(wr.ne.4) go to 140
      xp(kn)=xp(kn)-x(im2)+x(1)
      ip(kn)=ip(kn)-im2+1
      go to 130
  120 if(wt.le.2) go to 130
      if(wt.ne.4) go to 140
      yp(kn)=yp(kn)-y(jm2)+y(1)
      jp(kn)=jp(kn)-jm2+1
  130 kn=kn+1
      npn=npn+1
  140 k=k+1
      npt=npt+1
      pflg=1.0
      go to 10
  150 np=npn
      return
      end
