import networkx as nx

map_univ = [('Washington', 'Boise', 320), 
     ('Oregon', 'Boise', 400),
     ('Stanford', 'Boise', 600),
     ('Stanford', 'BYU', 650),
     ('Stanford', 'UNLV', 620),
     ('UCLA', 'UNLV', 266),
     ('UCLA', 'ASU', 560),
     ('Boise', 'Montana', 567),
     ('BYU', 'Montana' , 603),
     ('BYU', 'Wyoming', 190),
     ('BYU', 'USAF', 536),
     ('UNLV', 'USAF', 809),
     ('UNLV', 'NMST', 681),
     ('ASU', 'NMST', 279),
     ('Montana', 'Course', 320),
     ('Montana', 'Nebraska', 700),
     ('Wyoming', 'Course', 870),
     ('Wyoming', 'Nebraska', 236),
     ('Wyoming', 'OSU', 650),
     ('USAF', 'Nebraska', 256),
     ('USAF', 'OSU', 489),
     ('NMST', 'OSU', 200),
     ('NMST', 'UT', 546),
     ('Course', 'Carleton', 189),
     ('Course', 'Iowa', 369),
     ('Nebraska', 'Carleton', 740),
     ('Nebraska', 'Iowa', 106),
     ('Nebraska', 'Wash', 243),
     ('OSU', 'Iowa', 310),
     ('OSU', 'Wash', 165),
     ('OSU', 'Tulane', 670),
     ('UT', 'Wash', 840),
     ('UT', 'Tulane', 512),
     ('Carleton', 'Wisconsin', 380),
     ('Carleton', 'Illinois', 410),
     ('Iowa', 'Wisconsin', 210),
     ('Iowa', 'Illinois', 105),
     ('Wash', 'Illinois', 140),
     ('Wash', 'Vandy', 176),
     ('Tulane', 'Vandy', 450),
     ('Tulane', 'Alabama', 260),
     ('Wisconsin', 'Rochester', 210),
     ('Wisconsin', 'Dayton', 279),
     ('Illinois', 'Rochester', 310),
     ('Illinois', 'Dayton', 100),
     ('Vandy', 'Dayton', 200),
     ('Vandy', 'Gatech', 104),
     ('Alabama', 'Gatech', 90),
     ('Alabama', 'FSU', 400),
     ('Rochester', 'Brown', 620),
     ('Rochester', 'MIT', 540),
     ('Dayton', 'MIT', 689),
     ('Dayton', 'Georgetown', 210),
     ('Dayton', 'Duke', 311),
     ('Gatech', 'Duke', 140),
     ('FSU', 'Duke', 687)]
#b = 'Orgeon'
#c = 'Stan
#c = 
length = 140

G = nx.Graph()
for i in range(len(map_univ)):
    G.add_weighted_edges_from([map_univ[i]]) 
	
nx.draw(G, with_labels = True)

edge_labels = nx.get_edge_attributes(G, 'weight')
weight = G.get_edge_data('Washington', 'Boise')['weight']

print("edge_labels = ", weight)
        
