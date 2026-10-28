from database import create_table, get_all_persons


create_table()

persons = get_all_persons()

print("\nDANH SACH VAN TAY")

if len(persons) == 0:

    print("Database dang trong.")

else:

    for person in persons:

        print("-------------------------")
        print("ID:", person["id"])
        print("Ma:", person["person_code"])
        print("Ten:", person["name"])
        print("So minutiae:", len(person["template"]))
        print("Ngay dang ky:", person["created_at"])