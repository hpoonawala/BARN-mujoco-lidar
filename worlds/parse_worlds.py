import csv
import xml.etree.ElementTree as ET

def csv_to_xml(csv_filename, xml_filename):
    # Create the root XML element
    root = ET.Element("Data")

    # Open the CSV file and read rows
    with open(csv_filename, newline='\n') as csvfile:
        reader = csv.reader(csvfile)
        
        # Assuming the first row is the header
        count=-1;
        for row in reader:
            count+=1
            print(row,row[0],row[1])
            # for ent in range(2):
                # print(row[ent])
            # Create an XML element for each row
            # row_elem = ET.Element("geom name=\"unit_cylinder_"+str(count)+"\" type=\"cylinder\" pos=\""+row[0]+ " "+row[1]+" 0.0\" size=\"1 4\"  rgba=\"0.2 0.8 0.2 1\" ")
            row_elem = ET.Element("geom name=\"unit_cylinder_"+str(count)+"\" type=\"cylinder\" pos=\""+str(2*float(row[0])+3)+ " "+str(2*float(row[1])-3)+" 0.0\" size=\"0.2 4\"  rgba=\"0.2 0.8 0.2 1\" ")
            
            
            # Add the row to the root element
            root.append(row_elem)
    
    # Write the XML to a file
    # with open(xml_filename, "wb") as f:
        # for row_elem in root:
        # Convert each <Row> element to XML and write it individually
            # f.write(ET.tostring(row_elem, encoding="utf-8"))
            # f.write(b"\n")  # Add newline for better readability
    tree = ET.ElementTree(root)
    tree.write(xml_filename, encoding="utf-8", xml_declaration=True)
    print(f"XML file '{xml_filename}' has been created.")

# Example usage
for i in range(300):
    csv_to_xml("worlds/world_"+str(i)+".csv", "worlds/world_"+str(i)+".xml")

