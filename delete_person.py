
from database import get_all_persons, delete_person

persons = get_all_persons()

print("Danh sach nguoi da dang ky:")

for person in persons:
    print(
        "Ma:", person["person_code"],
        "| Ten:", person["name"],
        "| ID:", person["id"]
    )

person_code = input("\nNhap ma nguoi can xoa: ")

for person in persons:
    if person["person_code"] == person_code:
        confirm = input(
            f"Ban co chac muon xoa {person['name']}? (y/n): "
        )

        if confirm.lower() == "y":
            delete_person(person["id"])
            print("Da xoa thanh cong!")
        else:
            print("Da huy thao tac.")

        break
else:
    print("Khong tim thay ma nguoi nay!")
