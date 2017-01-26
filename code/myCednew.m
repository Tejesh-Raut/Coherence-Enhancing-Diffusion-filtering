function myCEDnew( )

myalpha = 0.001; sigma=0.7; T = 15; rho = 4; C= 1;
im = imread('../images/2.png');
[numrow, numcol] = size(im);
imorig = im;
stepT=0.15;
t = 0;
im=double(im);

while (t < (T-0.001))
    t = t + stepT;
    %% 1 gaussian K_sigma
    limitX=-ceil(2*sigma):ceil(2*sigma);
    kSigma = exp(-(limitX.^2/(2*sigma^2)));
    kSigma = kSigma/sum(kSigma(:));
    usigma=imfilter(imfilter(im,(kSigma'), 'same' ,'replicate'),kSigma, 'same' ,'replicate');
    
    %% Gradient
    [uy,ux]=gradient(usigma);
    
    %% 3 gaussian K_rho
    limitXJ=-ceil(3*rho):ceil(3*rho);
    kSigmaJ = exp(-(limitXJ.^2/(2*rho^2)));
    kSigmaJ = kSigmaJ/sum(kSigmaJ(:));
    Jxx = imfilter(imfilter((ux.^2),(kSigmaJ'), 'same' ,'replicate'),kSigmaJ, 'same' ,'replicate');
    Jxy = imfilter(imfilter((ux.*uy),(kSigmaJ'), 'same' ,'replicate'),kSigmaJ, 'same' ,'replicate');
    Jyy = imfilter(imfilter((uy.^2),(kSigmaJ'), 'same' ,'replicate'),kSigmaJ, 'same' ,'replicate');
    
    %% Principal axis transformation
    % Eigenvectors of J, v1 and v2
    v2x = zeros(numrow, numcol);
    v2y = zeros(numrow, numcol);
    lambda1 = zeros(numrow, numcol);
    lambda2 = zeros(numrow, numcol);
    for i=1:numrow
        for j=1:numcol
            pixel = [Jxx(i,j), Jxy(i,j); Jxy(i,j), Jyy(i,j)];
            [pixelV, pixelD] = eig(pixel);
            v2x(i,j) = pixelV(1,2);
            v2y(i,j) = pixelV(2,2);
            lambda1(i,j) = pixelD(1,1);
            lambda2(i,j) = pixelD(2,2);
            if((v2x(i,j)^2)+(v2y(i,j)^2)==0)
                abcd=0;
            else
                v2x(i,j) = v2x(i,j)/(sqrt((v2x(i,j)^2)+(v2y(i,j)^2)));
                v2y(i,j) = v2y(i,j)/(sqrt((v2x(i,j)^2)+(v2y(i,j)^2)));
            end;
        end;
    end;
    v1x = -v2y; 
    v1y = v2x;    
    
    %% Calculation of diffusion matrix
    di=(lambda1-lambda2);
    lambda1 = myalpha + (1 - myalpha)*exp(-C./(di).^(2)); 
    lambda2 = myalpha;
    
    Dxx = lambda1.*v1x.^2   + lambda2.*v2x.^2;
    Dxy = lambda1.*v1x.*v1y + lambda2.*v2x.*v2y;
    Dyy = lambda1.*v1y.^2   + lambda2.*v2y.^2;
    %% Non negativity discretization scheme referred from http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=16CCEAD1A72E6A1CC960DF99795D62B7?doi=10.1.1.21.632&rep=rep1&type=pdf
    im=non_negativity_discretization(im,Dxx,Dxy,Dyy,stepT);

end;

%% output
figure(1);
subplot(1, 2, 1);
imagesc(imorig);
title('Original image');
colormap('Gray');
daspect ([1 1 1]);

subplot(1, 2, 2);
imagesc(im);
title('Coherence Enhancing Diffusion Filtering');
colormap('Gray');
daspect ([1 1 1]);

figure(2);

subplot(1, 2, 1);
imagesc(imorig);
title('Original image');
colormap('Gray');
daspect ([1 1 1]);

subplot(1, 2, 2);
imagesc(atan2(double(uy), double(ux)));
title('Orientation of smooth gradient');
colormap('Gray');
daspect ([1 1 1]);

imwrite((uint8(im)), '../images/2CED.png');
end


function im=non_negativity_discretization(im,Dxx,Dxy,Dyy,stepT)
    % Make positive and negative indices
    [numrow,numcol] = size(im);
    px = [2:numrow,numrow]; nx = [1,1:numrow-1];
    py = [2:numcol,numcol]; ny = [1,1:numcol-1];

    % In literature a,b and c are used as variables 
    a=Dxx;b=Dxy;c=Dyy;

    % Stencil Weights
    wbR1 = (0.25)*((abs(b(nx, py))-b(nx,py)) + (abs(b)-b));
    wtM2 = (0.5)*( (c(:, py)+c) -(abs(b(:,py))+abs(b)));
    wbL3 = (0.25)*((abs(b(px, py))+b(px,py)) + (abs(b)+b));
    wmR4 = (0.5)*( (a(nx,: )+a) -(abs(b(nx,:))+abs(b)));
    wmL6 = (0.5)*( (a(px,: )+a) -(abs(b(px,:))+abs(b)));
    wtR7 = (0.25)*((abs(b(nx, ny))+b(nx,ny)) + (abs(b)+b));
    wmB8 = (0.5)*( (c(:, ny)+c) -(abs(b(:,ny))+abs(b)));
    wtL9 = (0.25)*((abs(b(px, ny))-b(px,ny)) + (abs(b)-b));
    im=  im+stepT*(wbR1.*(im(nx,py) -im(:,:))+wtM2.*(im(:, py) -im(:,:))+wbL3.*(im(px,py) -im(:,:))+wmR4.*(im(nx,:)  -im(:,:))+ ...      
                  wmL6.*(im(px,:)  -im(:,:))+ wtR7.*(im(nx,ny) -im(:,:))+ wmB8.*(im(:, ny) -im(:,:))+ wtL9.*(im(px,ny) -im(:,:)));
end