def pytest_addoption(parser):
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="enable longrundecorated tests",
    )

    parser.addoption(
        "--totake",
        type=int,
        required=False,
        default=-1,
        help="Images to take (default all)",
    )
