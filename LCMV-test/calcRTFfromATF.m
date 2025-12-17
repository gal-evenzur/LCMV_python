%% =========================================================================
%% Function calcRTFfromATF - computing the RTF based on given ATF 
%% =========================================================================
function  [ hkSteeredRtf ] = calcRTFfromATF( hSourceAroundMic , N , refMic , numMics , numSteered , epsilon )

    hkSteeredRtf = zeros(numSteered,N,numMics);
    for i = 1 : numSteered
        hkSteeredMics =fft( squeeze( hSourceAroundMic( i , : , : ) ) ) ;
        hkRefMic = hkSteeredMics( : , refMic ) ; 
        hkRefMicNorm = hkRefMic.*conj(hkRefMic) ;
        hkRefMicNorm = max(hkRefMicNorm , epsilon) ;
        
        for m = 1 : numMics
            %hkSteeredRtf( i , : , m )  = ifftshift( ifft (hkRefMic ./ ( hkSteeredMics( : ,  m ) + epsilon ) ) );            
            %hkSteeredRtf( i , : , m )  =  ifftshift(ifft ( ( hkSteeredMics( : ,  m ).*conj(hkRefMic) ) ./ ( hkRefMicNorm ) ) ) ;
            hnSteeredRtf  =  ifftshift(ifft ( ( hkSteeredMics( : ,  m ).*conj(hkRefMic) ) ./ ( hkRefMicNorm ) ) ) ;
            hnSteeredRtf( 7*N/8+1 : end) = zeros(N/8,1) ;  % Zeroing tail
            hnSteeredRtf( 1: N*3/8 ) = zeros(N*3/8,1);  % Zeroing head
            hkSteeredRtf( i , : , m ) = circshift( hnSteeredRtf , -N*(3/8) )  ;
       end
    end

    
    