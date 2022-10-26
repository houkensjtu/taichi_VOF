      subroutine prtplt (n)
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
c     *** print and plot
c     *** provides formatted writes to paper and film
c
      go to (10,70,90,130), n
c
c     *** prtplt (1) write out initial data and mesh data
c
   10 write (6,170)
      write (6,180) name
      write (6,220) ibar,jbar,delt,nu,icyl,epsi,gx,gy,ui,vi,velmx,twfin
     1 ,prtdt,pltdt,omg,alpha,wl,wr,wt,wb,imovy,autot,flht,isymplt,sigma
     2 ,isurf10,cangle,csq,nmat,rhof,rhofc
      if (imovy.gt.0) go to 40
      write (12,170)
      write (12,180) name
      write (12,220) ibar,jbar,delt,nu,icyl,epsi,gx,gy,ui,vi,velmx,twfin
     1 ,prtdt,pltdt,omg,alpha,wl,wr,wt,wb,imovy,autot,flht,isymplt,sigma
     2 ,isurf10,cangle,csq,nmat,rhof,rhofc
c
c     *** write on film variable mesh input data
c
      write (12,260) nkx
      do 20 i=1,nkx
      write (12,270) i,xl(i),xc(i),xl(i+1),nxl(i),nxr(i),dxmn(i)
   20 continue
      write (12,280) nky
      do 30 i=1,nky
      write (12,275) i,yl(i),yc(i),yl(i+1),nyl(i),nyr(i),dymn(i)
   30 continue
   40 continue
c
c     *** print variable mesh input data
c
      write (6,260) nkx
      do 50 i=1,nkx
      write (6,270) i,xl(i),xc(i),xl(i+1),nxl(i),nxr(i),dxmn(i)
   50 continue
      write (6,280) nky
      do 60 i=1,nky
      write (6,275) i,yl(i),yc(i),yl(i+1),nyl(i),nyr(i),dymn(i)
   60 continue
      go to 160
c
c     *** prtplt (2)  write time step, cycle information
c
   70 continue
      write (6,210) iter,t,delt,cycle,vchgt
      if (imovy.eq.1) go to 80
      if (t.gt.0.) go to 80
      write (12,210) iter,t,delt,cycle,vchgt
   80 continue
      go to 160
c
c     *** prtplt (3)  write field variables to film
c
   90 if (imovy.eq.1) go to 120
C      call adv (1)
      write (12,250) name
      write (12,210) iter,t,delt,cycle,vchgt
      write (12,240)
      write (12,290) nreg
      write (12,300)
      knr=nreg+5
      do 100 k=6,knr
      write (12,310) k,vol(k),pr(k)
  100 continue
      write (12,190)
      do 110 i=1,imax
      do 110 j=1,jmax
      dij=rdx(i)*(u(i,j)-u(i-1,j))+rdy(j)*(v(i,j)-v(i,j-1))+cyl*0.5*rxi
     1 (i)*(u(i,j)+u(i-1,j))
      write (12,200) i,j,u(i,j),v(i,j),p(i,j),dij,ps(i,j),f(i,j),nf(i,j)
     1 ,peta(i,j)
  110 continue
  120 continue
      go to 160
c
c     *** prtplt (4)  write field variables to paper
c
  130 write (6,170)
      write (6,250) name
      write (6,210) iter,t,delt,cycle,vchgt
      write (6,240)
      write (6,290) nreg
      write (6,300)
      knr=nreg+5
      do 140 k=6,knr
      write (6,310) k,vol(k),pr(k)
  140 continue
      write (6,240)
      write (6,190)
      do 150 i=1,imax
      do 150 j=1,jmax
      dij=rdx(i)*(u(i,j)-u(i-1,j))+rdy(j)*(v(i,j)-v(i,j-1))+cyl*0.5*rxi
     1 (i)*(u(i,j)+u(i-1,j))
      write (6,200) i,j,u(i,j),v(i,j),p(i,j),dij,ps(i,j),f(i,j),nf(i,j)
     1 ,peta(i,j)
  150 continue
  160 return
c
  170 format (1h1)
  180 format (10a8)
  190 format (4x,1hi,5x,1hj,9x,1hu,14x,1hv,15x,1hp,15x,1hd,12x,2hps,13x,
     1 1hf,11x,2hnf,9x,4hpeta)
  200 format (2x,i3,3x,i3,6(3x,1pe12.5),3x,i3,3x,e12.5)
  210 format (6x,6hiter= ,i5,5x,6htime= ,1pe12.5,5x,6hdelt= ,1pe12.5,5x,
     1 7hcycle= ,i4,5x,7hvchgt= ,1pe12.5)
  220 format (1h ,5x,6hibar= ,i3/6x,6hjbar= ,i3/6x,6hdelt= ,1pe12.5/8x,4
     1 hnu= ,e12.5/6x,6hicyl= ,i2/6x,6hepsi= ,e12.5/8x,4hgx= ,e12.5/8x,4
     2 hgy= ,e12.5/8x,4hui= ,e12.5/8x,4hvi= ,e12.5/5x,7hvelmx= ,e12.5/5x
     3 ,7htwfin= ,e12.5/5x,7hprtdt= ,e12.5/5x,7hpltdt= ,e12.5/7x,5homg=
     4 ,e12.5/5x,7halpha= ,e12.5/8x,4hwl= ,i2/8x,4hwr= ,i2/8x,4hwt= ,i2/
     5 8x,4hwb= ,i2/5x,7himovy= ,e12.5/5x,7hautot= ,e12.5/6x,6hflht=
     6 ,e12.5/3x,9hisymplt= ,i2/5x,7hsigma= ,e12.5/3x,9hisurf10= ,i2/4x,
     7 8hcangle= ,e12.5/7x,5hcsq= ,e12.5/6x,6hnmat= ,i2/6x,6hrhof= ,e12.
     8 5/,5x,7hrhofc= ,e12.5/)
  230 format (6x,6hcwtd= ,e12.5/6x,6htrst= ,e12.5/)
  240 format (1h0)
  250 format (1h ,18x,10a8,1x,a10,2(1x,a8))
  260 format (2x,5hnkx= ,i4)
  270 format(2x,8hmesh-x= ,i4,3x,4hxl= ,1pe12.5,3x,4hxc= ,e12.5,3x,
     1 4hxr= ,e12.5,3x,5hnxl= ,i4,3x,5hnxr= ,i4,3x,6hdxmn= ,e12.5)
  275 format(2x,8hmesh-y= ,i4,3x,4hyl= ,1pe12.5,3x,4hyc= ,e12.5,3x,
     1 4hyr= ,e12.5,3x,5hnyl= ,i4,3x,5hnyr= ,i4,3x,6hdymn= ,e12.5)
  280 format (2x,5hnky= ,i4)
  290 format (2x,6hnreg= ,i4)
  300 format (15x,1hk,6x,6hvol(k),9x,5hpr(k))
  310 format (13x,i3,2x,1pe12.5,3x,e12.5)
      end
