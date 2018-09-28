./baseline.py simple rnn based with query + passage one rnn encoding  
./qcatt.py  query passage separately encode and interact using attention same as rnet without using self match attention  
./rnet.py  query passage separately encode and interact using attention then add self match attention(support gate combine and sfu combine)  
./m_reader.py same as rnet but can have multi hop attention steps 
