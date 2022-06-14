#!/bin/bash

#Download Waterloo1k directory
wget -L https://utexas.box.com/shared/static/k1eerw4kfv8v8uzvxhkb3mpcsvi66y97 -O training_data/Waterloo1k_aa.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/qugg2e8rh9rcxlwk2oml3dxaellsebd7 -O training_data/Waterloo1k_ab.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/9fquu19a5ysx5m6zf2luhapz48vwa0jj -O training_data/Waterloo1k_ac.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/wrt8dvlk6lknkqvco8ocv7bmh34zgo75 -O training_data/Waterloo1k_ad.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/xna355b0lpr3qlzeqtv0u4gqo7mw5aws -O training_data/Waterloo1k_ae.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/bjadyqd2rudu5afiscn08vm8md0y1cfv -O training_data/Waterloo1k_af.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/vovv6u7rcp8um52bl2aq1kjxnves21ep -O training_data/Waterloo1k_ag.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/gz9twwsvn3r62ilhrqtsi0ew0qe1te2e -O training_data/Waterloo1k_ah.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/x1aom92stivg7iexs62dhnx5eljul1ez -O training_data/Waterloo1k_ai.zip -q --show-progress
cat training_data/Waterloo1k_* > training_data/Waterloo1k.zip
rm -f training_data/Waterloo1k_*

#Download REDs directory
wget -L https://utexas.box.com/shared/static/z2c2oufm8qdy6qey5hc8s64bjnyba6hz -O training_data/REDS_aa.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/b425i9fs9hk5o8rkg6i63wy7rmzy8xk0 -O training_data/REDS_ab.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/txbaf6fv3pszdk4oymxo8b6hq3ai7ny0 -O training_data/REDS_ac.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/603b5g1v16l07gxcegkvoyqe8r2ya2xb -O training_data/REDS_ad.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/u25pldqucozb8vyh0g6v3e1wqjlxnqyq -O training_data/REDS_ae.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/soedh5w6te5alx04u0fzn96lbq4xce1h -O training_data/REDS_af.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/78ma8l45akp0bzeyo40foc822frveyo5 -O training_data/REDS_ag.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/ptabanwvqyi7cbheh5fcs2ldejkieehd -O training_data/REDS_ah.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/kw569pa3umhnoohy0b8scjbb5uic6y6u -O training_data/REDS_ai.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/7v7sgtclulpr87elzu5mqpip3jjo5ash -O training_data/REDS_aj.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/tf5mfe9lgtqht6m0fqapbkmmg4wb682i -O training_data/REDS_ak.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/lgicp39q12zqo4apb7321761me4bpiz4 -O training_data/REDS_al.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/687abgp51h54wzf0xy8r06ksfwb5yln5 -O training_data/REDS_am.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/j9exrex72r8ili6u5h2ewsl9r7u009os -O training_data/REDS_an.zip -q --show-progress
cat training_data/REDS_* > training_data/REDS.zip
rm -f training_data/REDS_*

#Download UVG directory
wget -L https://utexas.box.com/shared/static/hg8w5w6kb6m8exzydm2w8rnve74dh70l.zip -O training_data/UVG.zip -q --show-progress

#Download MCML directory
wget -L https://utexas.box.com/shared/static/qs3rte5a0eq342qy82us06428t79igfj.zip -O training_data/MCML.zip -q --show-progress

#Download dareful directory
wget -L https://utexas.box.com/shared/static/xkhm1scirbin1dmz7f45t2tikgoslke8 -O training_data/dareful_aa.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/aiwzplg1p5h0axxxiw1a5zdkydfprm1i -O training_data/dareful_ab.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/fqqnx21bsumlsib15cjr5ifg71rzvz17 -O training_data/dareful_ac.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/u1upzahh8or4edclhl8tder4di789fgw -O training_data/dareful_ad.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/zvuwx8e6ptb4y55fhrv3t3ljhgmmx8ga -O training_data/dareful_ae.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/89gy3gicdbllxduznqg9osekc3msq184 -O training_data/dareful_af.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/13oxybjwcm5igoi9k7abqc167d7cz42m -O training_data/dareful_ag.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/ojpvuvj3ru8ot3ftxlp46x6dcl4u773b -O training_data/dareful_ah.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/tn77ve7n423dcnesii47o9s2bimj6g2d -O training_data/dareful_ai.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/ao7ooqvnnw8tihhwgkqbyr67uunygd32 -O training_data/dareful_aj.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/faq3nz4ov78cawwcwo3ecr506k3e24bz -O training_data/dareful_ak.zip -q --show-progress
wget -L https://utexas.box.com/shared/static/q0ex7acbgyoantj8lk1tn3ihbxkeetus -O training_data/dareful_al.zip -q --show-progress
cat training_data/dareful_* > training_data/dareful.zip
rm -f training_data/dareful_*
